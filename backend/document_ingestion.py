import os
import json
import time
import hashlib

from dotenv import load_dotenv

from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# ---- Config ----
PDF_STRATEGY = "auto"         
CHUNK_COMBINE_UNDER = 300   
PERSIST_DIR = "./chroma_db"
EMBED_BATCH_SIZE = 100


def partition_document(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found at: {file_path}")

    print(f"Partitioning '{file_path}' with strategy='{PDF_STRATEGY}'...")
    start = time.time()

    try:
        elements = partition_pdf(
            filename=file_path,
            strategy=PDF_STRATEGY,
            infer_table_structure=True,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to partition PDF '{file_path}': {e}") from e

    elapsed = time.time() - start
    print(f"Partitioned in {elapsed:.1f}s — Elements: {len(elements)}")

    for e in elements[:20]:
        print(type(e).__name__)

    return elements


def chunk_document(elements):
    chunks = chunk_by_title(
        elements=elements,
        max_characters=CHUNK_MAX_CHARACTERS,
        combine_text_under_n_chars=CHUNK_COMBINE_UNDER,
    )
    return chunks


def separate_content(chunk):
    """
    Splits a chunk into narrative text (with table text removed, to avoid
    duplicating it) and a list of tables rendered as HTML where possible.
    """

    data = {
        "text_parts": [],
        "tables": [],
        "pages": set(),
    }

    if hasattr(chunk.metadata, "orig_elements") and chunk.metadata.orig_elements:
        for item in chunk.metadata.orig_elements:
            kind = type(item).__name__

            page_number = getattr(item.metadata, "page_number", None)
            if page_number is not None:
                data["pages"].add(page_number)

            if kind == "Table":
                table_html = getattr(item.metadata, "text_as_html", None)
                data["tables"].append(table_html if table_html else item.text)
            else:
                if item.text:
                    data["text_parts"].append(item.text)
    else:
        # Fallback: no orig_elements available, use chunk text as-is
        data["text_parts"].append(chunk.text)

    data["text"] = "\n".join(data["text_parts"])
    return data


def process_chunks(chunks, source_name):
    documents = []

    print("Total chunks:", len(chunks))

    for i, chunk in enumerate(chunks):
        data = separate_content(chunk)

        text = data["text"]
        tables = data["tables"]
        pages = sorted(data["pages"])

        if tables:
            combined_content = f"{text}\n\nTABLES:\n{' '.join(tables)}".strip()
        else:
            combined_content = text.strip()

        if len(combined_content) < 10:
            print(f"Skipping empty chunk {i}")
            continue

        doc = Document(
            page_content=combined_content,
            metadata={
                "raw_text": text,
                "tables": json.dumps(tables),
                "source": source_name,
                "pages": ",".join(str(p) for p in pages) if pages else "unknown",
                "chunk_index": i,
            },
        )

        documents.append(doc)

    print("Processed docs:", len(documents))
    return documents


def make_doc_id(doc: Document) -> str:
    """Deterministic ID from content + source, so re-running ingestion
    on the same PDF upserts instead of duplicating vectors."""
    key = f"{doc.metadata.get('source')}::{doc.page_content}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def create_vectorstore(documents):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
    )

    ids = [make_doc_id(doc) for doc in documents]

    print(f"Inserting {len(documents)} documents in batches of {EMBED_BATCH_SIZE}...")

    for start in range(0, len(documents), EMBED_BATCH_SIZE):
        end = start + EMBED_BATCH_SIZE
        batch_docs = documents[start:end]
        batch_ids = ids[start:end]

        vectorstore.add_documents(documents=batch_docs, ids=batch_ids)
        print(f"  Inserted {min(end, len(documents))}/{len(documents)}")

    return vectorstore


if __name__ == "__main__":
    file_path = os.path.join("docs", "rag.pdf")
    source_name = os.path.basename(file_path)

    try:
        print("Loading PDF...")
        elements = partition_document(file_path)

        print("Creating chunks...")
        chunks = chunk_document(elements)

        print("Processing chunks...")
        processed_documents = process_chunks(chunks, source_name)

        if not processed_documents:
            print("No documents produced from this PDF — nothing to ingest.")
        else:
            print("Generating embeddings and storing vectors...")
            vectorstore = create_vectorstore(processed_documents)
            print("Ingestion completed")

    except FileNotFoundError as e:
        print(f"File error: {e}")
    except RuntimeError as e:
        print(f"Processing error: {e}")
    except Exception as e:
        print(f"Unexpected error during ingestion: {e}")
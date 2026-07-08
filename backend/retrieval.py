import os

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
TOP_K = 5


def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
    )

    print(f"\nTotal chunks in database: {vectorstore._collection.count()}\n")

    return vectorstore


def retrieve_chunks(vectorstore, query, k=TOP_K):
    results = vectorstore.similarity_search_with_score(query, k=k)

    print("=" * 80)
    print("Retrieved Chunks")
    print("=" * 80)

    for i, (doc, score) in enumerate(results, start=1):
        print(f"\nChunk {i}")
        print(f"Distance: {score:.4f}")
        print("-" * 40)
        print(doc.page_content[:500])
        print("-" * 40)

    return results


def build_context(results):
    context = []

    for i, (doc, score) in enumerate(results, start=1):
        source = doc.metadata.get("source", "unknown")

        context.append(
            f"[Source {i} - {source} | distance={score:.4f}]\n{doc.page_content}"
        )

    return "\n\n".join(context)


def build_prompt(context, query, history):
    history_text = ""

    if history:
        previous = []

        for q, a in history[-3:]:
            previous.append(f"Q: {q}\nA: {a}")

        history_text = (
            "PREVIOUS CONVERSATION:\n"
            + "\n\n".join(previous)
            + "\n\n"
        )

    return f"""
You are an academic assistant.

Answer ONLY using the provided context.

Rules:

1. Use only the provided context.
2. Never use outside knowledge.
3. Cite sources like [Source 1].
4. If the answer is not present, reply exactly:
Information not found in notes.
5. Preserve technical terminology.

{history_text}

CONTEXT:

{context}

QUESTION:

{query}

ANSWER:
"""


def main():

    vectorstore = load_vectorstore()

    llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
    )

    history = []

    while True:

        query = input("\nAsk Question: ").strip()

        if query.lower() in ["exit", "quit"]:
            break

        if not query:
            continue

        results = retrieve_chunks(vectorstore, query)

        if len(results) == 0:
            print("Information not found in notes.")
            continue

        context = build_context(results)

        prompt = build_prompt(
            context,
            query,
            history,
        )

        print("\nAnswer:\n")

        answer = ""

        for chunk in llm.stream(prompt):
            print(chunk.content, end="", flush=True)
            answer += chunk.content

        print("\n")

        history.append((query, answer))


if __name__ == "__main__":
    main()
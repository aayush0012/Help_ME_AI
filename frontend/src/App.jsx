import { useState, useEffect, useRef } from "react";
import axios from "axios";
import "./App.css";

function App() {

  const [file, setFile] = useState(null);
  const [message, setMessage] = useState("");

  const [input, setInput] = useState("");

  const [messages, setMessages] = useState([]);

  const [uploading, setUploading] = useState(false);

  const [thinking, setThinking] = useState(false);

  const [typingText, setTypingText] = useState("");

  const [displayedTyping, setDisplayedTyping] = useState("");

  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({
      behavior: "smooth",
    });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, displayedTyping, thinking]);

  useEffect(() => {
    if (!typingText) {
      setDisplayedTyping("");
      return;
    }

    let i = 0;

    const interval = setInterval(() => {
      setDisplayedTyping(
        typingText.slice(0, i + 1)
      );

      i++;

      if (i >= typingText.length) {

        clearInterval(interval);

        setMessages((prev) => [
          ...prev,
          {
            id: Date.now(),
            sender: "bot",
            text: typingText,
          },
        ]);

        setTypingText("");
        setDisplayedTyping("");
      }

    }, 15);

    return () => clearInterval(interval);

  }, [typingText]);

  const uploadFile = async () => {

    if (!file) return;

    setUploading(true);

    const formData = new FormData();

    formData.append("file", file);

    try {

      const response = await axios.post(
        "http://127.0.0.1:8000/upload",
        formData
      );

      setMessage(response.data.message);

    } catch (err) {

      console.log(err);

      const detail = err.response?.data?.detail;

      setMessage(
        detail
          ? `Upload failed : ${detail}`
          : "Upload Failed"
      );

    }

    setUploading(false);

  };

  const handleSend = async () => {

    if (!input.trim()) return;

    const question = input;

    setMessages((prev) => [
      ...prev,
      {
        id: Date.now(),
        sender: "user",
        text: question,
      },
    ]);

    setInput("");

    setThinking(true);

    try {

      const response = await axios.post(
        `http://127.0.0.1:8000/chat?question=${encodeURIComponent(question)}`
      );

      setThinking(false);

      setTypingText(response.data.answer);

    } catch (err) {

      console.log(err);

      setThinking(false);

      const detail = err.response?.data?.detail;

      setTypingText(
        detail || "Unable to connect to server."
      );

    }

  };

  const handleKeyDown = (e) => {

    if (e.key === "Enter" && !e.shiftKey) {

      e.preventDefault();

      if (!thinking && !typingText) {
        handleSend();
      }

    }

  };
return (
  <div className="app">

    {/* Header */}

    <header className="header">

      <div className="logo">
         HelpMe AI
      </div>

      <div className="upload-area">

        <label className="upload-box">

          <input
            type="file"
            accept="application/pdf"
            onChange={(e) => setFile(e.target.files[0])}
          />

          <span>📄</span>

          <div>

            <h4>
              {file ? file.name : "Choose PDF"}
            </h4>

            <p>Click to browse</p>

          </div>

        </label>

        <button
          className="upload-btn"
          onClick={uploadFile}
          disabled={!file || uploading}
        >
          {uploading ? "Uploading..." : "Upload PDF"}
        </button>

      </div>

    </header>

    {message && (
      <div className="status">
        {message}
      </div>
    )}

    {/* Chat */}

    <div className="chat-container">

      {messages.length === 0 &&
        !thinking &&
        !typingText && (

          <div className="welcome">

            <div className="welcome-icon">
              🤖
            </div>

            <h1>Welcome to HelpMe AI</h1>

            <p>
              Upload a PDF and ask anything about
              your document.
            </p>

          </div>

      )}

      {messages.map((msg) => (

        <div
          key={msg.id}
          className={
            msg.sender === "user"
              ? "message user"
              : "message bot"
          }
        >

          <div className="avatar">
            {msg.sender === "user"
              ? "👤"
              : "🤖"}
          </div>

          <div className="bubble">

            {msg.text}

          </div>

        </div>

      ))}

      {thinking && (

        <div className="message bot">

          <div className="avatar">
            🤖
          </div>

          <div className="bubble">

            <div className="typing-indicator">

              <span></span>
              <span></span>
              <span></span>

            </div>

          </div>

        </div>

      )}

      {typingText && (

        <div className="message bot">

          <div className="avatar">
            🤖
          </div>

          <div className="bubble">

            {displayedTyping}

            <span className="cursor"></span>

          </div>

        </div>

      )}

      <div ref={messagesEndRef}></div>

    </div>

    {/* Bottom Input */}

    <div className="bottom">

      <input
        className="chat-input"
        type="text"
        placeholder="Ask anything about your document..."
        value={input}
        onChange={(e) => setInput(e.target.value)}
        onKeyDown={handleKeyDown}
        disabled={thinking || typingText !== ""}
      />

      <button
        className="send-btn"
        onClick={handleSend}
        disabled={
          !input.trim() ||
          thinking ||
          typingText !== ""
        }
      >
        ➜
      </button>

    </div>

  </div>
);
}
export default App;
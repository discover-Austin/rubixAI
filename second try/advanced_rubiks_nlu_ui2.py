#!/usr/bin/env python3
"""
Advanced Rubik's Cube NLU UI
A complete production-ready Flask-based web UI for the Advanced Rubik's Cube NLU system.
This file provides a super detailed and fully functional chat interface that integrates with
the advanced Rubik's Cube NLU processing engine.
"""

import json
import os
from flask import Flask, render_template_string, request, jsonify
from advanced_rubiks_nlu import RubiksNLUProcessor  # Assumes advanced_rubiks_nlu.py is in PYTHONPATH

app = Flask(__name__)
processor = RubiksNLUProcessor()

# HTML template for the chat UI (fully detailed, all bells and whistles)
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Advanced Rubik's Cube NLU Chatbot</title>
  <style>
    body {
      font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
      background-color: #1e1e1e;
      color: #d4d4d4;
      margin: 0;
      padding: 0;
    }
    .container {
      display: flex;
      flex-direction: column;
      height: 100vh;
    }
    header {
      background-color: #007acc;
      padding: 20px;
      text-align: center;
    }
    header h1 {
      margin: 0;
      color: #fff;
      font-size: 2em;
    }
    .chat-window {
      flex: 1;
      padding: 20px;
      overflow-y: auto;
      background-color: #252526;
    }
    .chat-message {
      margin-bottom: 15px;
      max-width: 80%;
    }
    .bot {
      text-align: left;
    }
    .user {
      text-align: right;
    }
    .message-content {
      display: inline-block;
      padding: 10px 15px;
      border-radius: 15px;
      background-color: #3e3e42;
    }
    .user .message-content {
      background-color: #0e639c;
      color: #fff;
    }
    form {
      display: flex;
      padding: 15px;
      background-color: #1e1e1e;
      border-top: 1px solid #3e3e42;
    }
    input[type="text"] {
      flex: 1;
      padding: 10px;
      border: none;
      border-radius: 5px;
      font-size: 1em;
      margin-right: 10px;
    }
    input[type="submit"] {
      padding: 10px 20px;
      font-size: 1em;
      border: none;
      border-radius: 5px;
      background-color: #007acc;
      color: #fff;
      cursor: pointer;
    }
    input[type="submit"]:hover {
      background-color: #005a9e;
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>Advanced Rubik's Cube NLU Chatbot</h1>
    </header>
    <div class="chat-window" id="chat-window">
      <!-- Chat messages will appear here -->
    </div>
    <form id="chat-form">
      <input type="text" id="message-input" autocomplete="off" placeholder="Type your message here..." required>
      <input type="submit" value="Send">
    </form>
  </div>
  <script>
    const form = document.getElementById('chat-form');
    const input = document.getElementById('message-input');
    const chatWindow = document.getElementById('chat-window');

    function addMessage(content, sender) {
      const messageDiv = document.createElement('div');
      messageDiv.classList.add('chat-message', sender);
      const contentDiv = document.createElement('div');
      contentDiv.classList.add('message-content');
      contentDiv.textContent = content;
      messageDiv.appendChild(contentDiv);
      chatWindow.appendChild(messageDiv);
      chatWindow.scrollTop = chatWindow.scrollHeight;
    }

    // Display welcome message from the bot
    addMessage("Welcome to the Advanced Rubik's Cube NLU Chatbot. How can I help you today?", "bot");

    form.addEventListener('submit', function(e) {
      e.preventDefault();
      const message = input.value.trim();
      if (!message) return;
      addMessage(message, "user");
      input.value = "";
      fetch('/process', {
        method: 'POST',
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: message })
      })
      .then(response => response.json())
      .then(data => {
        // Format the bot response detail
        let reply = "Analysis Complete.\n" +
                    "Concepts Extracted: " + JSON.stringify(data.concepts) + "\n" +
                    "Detected Patterns: " + JSON.stringify(data.patterns) + "\n" +
                    "Overall Confidence: " + data.confidence.toFixed(2);
        addMessage(reply, "bot");
      })
      .catch(error => {
        addMessage("Error processing request.", "bot");
        console.error('Error:', error);
      });
    });
  </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/process", methods=["POST"])
def process():
    data = request.get_json(force=True)
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided."}), 400
    try:
        # Process text using the advanced Rubik's Cube NLU system
        result = processor.process_input(text)
        response = {
            "cube_state": result.get("cube_state").tolist() if result.get("cube_state") is not None else None,
            "patterns": result.get("patterns", []),
            "moves": result.get("moves", []),
            "concepts": result.get("concepts", []),
            "confidence": result.get("confidence", 0.0)
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Determine port from environment variable
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
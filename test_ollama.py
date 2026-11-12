import ollama

client = ollama.Client(host="http://localhost:11434")

response = client.chat(
    model="llama3.1:8b",
    messages=[
        {
            "role": "user",
            "content": "What is FastAPI?"
        }
    ]
)

print(response["message"]["content"])
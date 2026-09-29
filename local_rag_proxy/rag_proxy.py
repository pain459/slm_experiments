# rag_proxy.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chromadb
import requests
import json

app = FastAPI()

# --- 1. Initialize ChromaDB (Local Vector Store) ---
# This stores data in a local folder named 'chroma_db'
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="python_curriculum")

# --- 2. Helper Functions ---
def get_embedding(text: str) -> list[float]:
    """Gets embedding from local Ollama nomic-embed-text model."""
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": "nomic-embed-text", "prompt": text}
    )
    return response.json()["embedding"]

def query_rag(query_text: str, n_results: int = 3) -> str:
    """Searches ChromaDB for relevant context."""
    query_embedding = get_embedding(query_text)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    
    if not results['documents'][0]:
        return ""
    
    # Combine retrieved documents into a single context string
    context = "\n\n---\n\n".join(results['documents'][0])
    return context

# --- 3. API Endpoints ---

class IngestRequest(BaseModel):
    doc_id: str
    text: str

@app.post("/ingest")
def ingest_document(req: IngestRequest):
    """Endpoint to add your curriculum data to the vector DB."""
    embedding = get_embedding(req.text)
    collection.upsert(
        ids=[req.doc_id],
        embeddings=[embedding],
        documents=[req.text]
    )
    return {"status": "success", "message": f"Document {req.doc_id} ingested."}

class ChatRequest(BaseModel):
    model: str
    messages: list[dict]
    stream: bool = False

@app.post("/v1/chat/completions")
def chat_completion(req: ChatRequest):
    """OpenAI-compatible endpoint for OpenCode to talk to."""
    # 1. Extract the user's latest prompt
    user_prompt = ""
    for msg in reversed(req.messages):
        if msg["role"] == "user":
            user_prompt = msg["content"]
            break
            
    if not user_prompt:
        raise HTTPException(status_code=400, detail="No user prompt found.")

    # 2. Query our local RAG for context
    rag_context = query_rag(user_prompt)
    
    # 3. Augment the prompt with RAG context
    system_prompt = "You are an expert Python coding assistant. Use the following context from the curriculum to answer the user's question accurately.\n\n"
    if rag_context:
        system_prompt += f"### RETRIEVED CONTEXT ###\n{rag_context}\n### END CONTEXT ###\n\n"
    else:
        system_prompt += "No specific context was retrieved. Rely on your general knowledge.\n\n"

    # 4. Rebuild the messages array with the augmented system prompt
    augmented_messages = [{"role": "system", "content": system_prompt}]
    augmented_messages.extend(req.messages)

    # 5. Forward to Ollama's OpenAI-compatible endpoint
    ollama_url = "http://localhost:11434/v1/chat/completions"
    payload = {
        "model": "qwen2.5:1.5b", # Placeholder until your custom GGUF is ready
        "messages": augmented_messages,
        "stream": req.stream
    }
    
    # Note: For simplicity, we are doing a non-streaming request here. 
    # OpenCode handles streaming, but let's get the base logic working first.
    payload["stream"] = False 
    
    response = requests.post(ollama_url, json=payload)
    
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Ollama error: {response.text}")
        
    return response.json()

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting RAG Proxy on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
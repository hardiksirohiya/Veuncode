import os
import io
import base64
import cv2
import uuid
from typing import Optional, List, Dict
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
import google.generativeai as genai
from chromadb.utils import embedding_functions
from chromadb import PersistentClient
import numpy as np
import uvicorn

# --- ChromaDB Setup ---
client = PersistentClient(path="./chroma_db")

from chromadb.utils.embedding_functions import GoogleGenerativeAiEmbeddingFunction

try:
    gemini_ef = GoogleGenerativeAiEmbeddingFunction(api_key=os.getenv("gemini_api_key"))
    chat_history_collection = client.get_or_create_collection(
        name="chat_history",
        embedding_function=gemini_ef
    )
except Exception as e:
    print(f"Error creating ChromaDB collection with Gemini embedding function: {e}")
    chat_history_collection = None


# --- FastAPI Setup ---
app = FastAPI(title="Multimodal Chat API")


# --- Gemini API Client Initialization ---
try:
    gemini_api_key = os.getenv("gemini_api_key") # Replace with your key
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY is not set.")
    genai.configure(api_key=gemini_api_key)
except ValueError as e:
    print(f"API key configuration error: {e}")


# --- Helper Functions ---
def extract_frames(video_path: str, fps: int = 1):
    """
    Extracts frames from a video at a specified frames-per-second (fps).
    Returns a list of image parts for the Gemini API.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file.")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    
    frame_interval = int(video_fps / fps) if video_fps > 0 else 1
    
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_interval == 0:
            is_success, buffer = cv2.imencode(".jpg", frame)
            if is_success:
                frames.append({
                    "inline_data": {
                        "mime_type": "image/jpeg", 
                        "data": base64.b64encode(buffer).decode()
                    }
                })
        
        count += 1
    
    cap.release()
    return frames

# --- API Endpoints ---

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Service is running"}

@app.post("/infer")
async def unified_chat(
    prompt: str = Form(...),
    video_file: Optional[UploadFile] = File(None),
    session_id: Optional[str] = Form(None)
):
    """
    This single endpoint handles both multimodal (video + text) and text-only chat.
    It processes a video if provided, otherwise, it handles a standard text query.
    """
    if not gemini_api_key:
        raise HTTPException(status_code=500, detail="Gemini API key is not set.")

    if not session_id:
        session_id = str(uuid.uuid4())

    context_history = ""
    if chat_history_collection:
        try:
            query_embedding = gemini_ef([prompt])[0]
            results = chat_history_collection.query(
                query_embeddings=[query_embedding],
                n_results=5,
                where={"session_id": session_id}
            )
            if results['documents'] and results['documents'][0]:
                for doc in reversed(results['documents'][0]):
                    context_history += f"Previous conversation: {doc}\n"
        except Exception as e:
            print(f"Error querying ChromaDB: {e}")

    frames = []
    if video_file:
        temp_video_path = f"temp_{video_file.filename}"
        with open(temp_video_path, "wb") as f:
            f.write(await video_file.read())

        frames = extract_frames(temp_video_path, fps=1)
        os.remove(temp_video_path)

        if not frames:
            raise HTTPException(status_code=400, detail="Could not extract frames from video.")
    
    prompt_with_history = f"{context_history}User's question: {prompt}"
    content = [{"role": "user", "parts": [{"text": prompt_with_history}] + frames}]

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(
            contents=content,
            generation_config=genai.GenerationConfig(max_output_tokens=500)
        )
        ai_response = response.text
        
        if chat_history_collection:
            chat_history_collection.add(
                documents=[f"User: {prompt}\nAssistant: {ai_response}"],
                metadatas=[{"session_id": session_id}],
                ids=[str(uuid.uuid4())]
            )

        return PlainTextResponse(content=ai_response)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API Error: {e}")


# The main block to run the application
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9525)

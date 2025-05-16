from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import openai
from pinecone import Pinecone, ServerlessSpec
import tempfile
import os
import uuid
import ffmpeg
import shutil

openai.api_key = "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"
pc = Pinecone(api_key="pcsk_475ix6_QNMj2etqYWbrUz2aKFQebCPzCepmZEsZFoWsMG3wjYvFaxdUFu73h7GWbieTeti")
pinecone_index = pc.Index("youtube-transcript")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def convert_to_mp3(input_path: str, output_path: str):
    ffmpeg.input(input_path).output(output_path, format='mp3').run(overwrite_output=True)

def get_embedding(text):
    response = openai.Embedding.create(input=text, model="text-embedding-3-small")
    return response['data'][0]['embedding']

def split_text(text, chunk_size=200):
    sentences = text.split('. ')
    chunks = []
    current = ''
    for s in sentences:
        if len(current) + len(s) < chunk_size:
            current += s + '. '
        else:
            chunks.append(current.strip())
            current = s + '. '
    if current:
        chunks.append(current.strip())
    return chunks

@app.post("/transcribe")
async def transcribe_and_embed(request: Request):
    try:
        # Read raw binary data from request
        body = await request.body()

        # Save to a temporary .mp4 file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(body)
            video_path = temp_video.name

        # Convert to mp3
        mp3_path = video_path.replace(".mp4", ".mp3")
        convert_to_mp3(video_path, mp3_path)

        # Transcribe using Whisper
        with open(mp3_path, "rb") as audio_file:
            transcript_response = openai.Audio.transcribe(
                model="whisper-1",
                file=audio_file
            )
        transcript_text = transcript_response['text']

        # Chunk and embed into Pinecone
        chunks = split_text(transcript_text)
        vectors = []
        for i, chunk in enumerate(chunks):
            vectors.append({
                "id": f"chunk-{uuid.uuid4()}",
                "values": get_embedding(chunk),
                "metadata": {"text": chunk}
            })

        pinecone_index.upsert(vectors)

        # Cleanup
        os.remove(video_path)
        os.remove(mp3_path)

        return JSONResponse(content={"transcript": transcript_text})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

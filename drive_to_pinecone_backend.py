from fastapi import FastAPI, File, UploadFile
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
async def transcribe_and_embed(file: UploadFile = File(...)):
    try:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            shutil.copyfileobj(file.file, temp_video)
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

        # Chunk and embed
        chunks = split_text(transcript_text)
        vectors = []
        for i, chunk in enumerate(chunks):
            vector = get_embedding(chunk)
            vectors.append({
                "id": f"chunk-{uuid.uuid4()}",
                "values": vector,
                "metadata": {"text": chunk}
            })

        pinecone_index.upsert(vectors)

        # Clean up temp files
        os.remove(video_path)
        os.remove(mp3_path)

        return JSONResponse(content={"transcript": transcript_text})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pinecone import Pinecone
from google.oauth2 import service_account
from googleapiclient.discovery import build
import requests
import tempfile
import os
import uuid
import ffmpeg
import glob
import sys

# === Environment Variables ===
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
gcred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

if not openai_api_key:
    raise RuntimeError("❌ OPENAI_API_KEY not found in environment.")
if not pinecone_api_key:
    raise RuntimeError("❌ PINECONE_API_KEY not found in environment.")
if not gcred_path or not os.path.exists(gcred_path):
    raise RuntimeError("❌ GOOGLE_APPLICATION_CREDENTIALS path is invalid or not set.")

print("✅ OPENAI_API_KEY prefix:", openai_api_key[:10])
print("✅ PINECONE_API_KEY prefix:", pinecone_api_key[:10])

client = OpenAI(api_key=openai_api_key)
pc = Pinecone(api_key=pinecone_api_key)
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
    print("🎛️ Converting to MP3...")
    ffmpeg.input(input_path).output(output_path, format='mp3').run(overwrite_output=True, quiet=True)
    print("✅ MP3 created at:", output_path)

def split_audio_to_chunks(mp3_path, chunk_folder):
    print("📼 Splitting MP3 into 5-min chunks...")
    os.makedirs(chunk_folder, exist_ok=True)
    output_pattern = os.path.join(chunk_folder, "chunk_%03d.mp3")
    ffmpeg.input(mp3_path).output(output_pattern, f='segment', segment_time=300, c='copy').run(overwrite_output=True, quiet=True)
    chunks = sorted(glob.glob(os.path.join(chunk_folder, "chunk_*.mp3")))
    print(f"✅ Total chunks created: {len(chunks)}")
    return chunks

def get_embedding(text):
    print("📐 Getting embedding for chunk of length:", len(text))
    response = client.embeddings.create(input=text, model="text-embedding-3-small")
    return response.data[0].embedding

def split_text(text, chunk_size=200):
    print("✂️ Splitting transcript into chunks...")
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
    print(f"✅ Total text chunks created: {len(chunks)}")
    return chunks

def download_video_from_drive(file_id, destination_path):
    print(f"🎯 Downloading video from Drive file_id={file_id}")
    credentials = service_account.Credentials.from_service_account_file(
        gcred_path,
        scopes=["https://www.googleapis.com/auth/drive.readonly"]
    )

    # ✅ Refresh token manually using a Request object
    from google.auth.transport.requests import Request
    credentials.refresh(Request())

    headers = {"Authorization": f"Bearer {credentials.token}"}
    url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media"

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        with open(destination_path, "wb") as f:
            f.write(response.content)
        print("✅ Downloaded video to:", destination_path)
    else:
        raise Exception(f"Download failed: {response.status_code} - {response.text}")


@app.post("/transcribe")
async def transcribe_and_embed(request: Request):
    try:
        print("🚀 /transcribe endpoint hit")
        data = await request.json()
        file_id = data.get("file_id")

        if not file_id:
            return JSONResponse(status_code=400, content={"error": "Missing file_id"})

        video_path = tempfile.mktemp(suffix=".mp4")
        download_video_from_drive(file_id, video_path)

        mp3_path = video_path.replace(".mp4", ".mp3")
        convert_to_mp3(video_path, mp3_path)

        chunk_folder = tempfile.mkdtemp()
        audio_chunks = split_audio_to_chunks(mp3_path, chunk_folder)

        full_transcript = ""
        for i, chunk_path in enumerate(audio_chunks):
            print(f"🧠 Transcribing chunk {i + 1}/{len(audio_chunks)}")
            with open(chunk_path, "rb") as chunk_file:
                result = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=chunk_file,
                    response_format="text"
                )
            full_transcript += result + "\n"

        print("✅ Full transcription completed. Preview:", full_transcript[:200])
        chunks = split_text(full_transcript)
        vectors = []

        print("📡 Uploading to Pinecone...")
        for chunk in chunks:
            vectors.append({
                "id": f"chunk-{uuid.uuid4()}",
                "values": get_embedding(chunk),
                "metadata": {"text": chunk}
            })
        pinecone_index.upsert(vectors)
        print("✅ Uploaded to Pinecone")

        os.remove(video_path)
        os.remove(mp3_path)
        for f in audio_chunks:
            os.remove(f)
        os.rmdir(chunk_folder)
        print("🧹 Cleaned all temporary files")

        return JSONResponse(content={"transcript": full_transcript.strip()})

    except Exception as e:
        print("❌ Error:", str(e))
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/embed")
async def embed_text(request: Request):
    try:
        data = await request.json()
        print("🔢 /embed endpoint called")
        embedding = get_embedding(data["text"])
        return {"embedding": embedding}
    except Exception as e:
        print("❌ Embed Error:", str(e))
        return JSONResponse(status_code=500, content={"error": str(e)})

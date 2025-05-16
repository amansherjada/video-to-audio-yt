from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pinecone import Pinecone
import tempfile
import os
import uuid
import ffmpeg
import glob  # 🔄
import sys

# === Environment Variables ===
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

if not openai_api_key:
    raise RuntimeError("❌ OPENAI_API_KEY not found in environment.")
if not pinecone_api_key:
    raise RuntimeError("❌ PINECONE_API_KEY not found in environment.")

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

# 🔄 Split MP3 into 5-minute chunks
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

@app.post("/transcribe")
async def transcribe_and_embed(request: Request):
    try:
        print("🚀 /transcribe endpoint hit")
        body = await request.body()
        print("📦 Received video payload of size:", len(body), "bytes")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(body)
            video_path = temp_video.name
        print("📝 Saved .mp4 to:", video_path)

        mp3_path = video_path.replace(".mp4", ".mp3")
        convert_to_mp3(video_path, mp3_path)

        # 🔄 SPLIT AUDIO INTO CHUNKS
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
        print("🧹 All temporary files cleaned")

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

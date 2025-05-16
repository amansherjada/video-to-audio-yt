from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pinecone import Pinecone
import tempfile
import os
import uuid
import ffmpeg

# Load API keys from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

if not openai_api_key:
    raise RuntimeError("❌ OPENAI_API_KEY not found in environment.")
if not pinecone_api_key:
    raise RuntimeError("❌ PINECONE_API_KEY not found in environment.")

client = OpenAI(api_key=openai_api_key)
pc = Pinecone(api_key=pinecone_api_key)
pinecone_index = pc.Index("youtube-transcript")  # Change if your index name is different

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, restrict to your GAS domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def convert_to_mp3(input_path: str, output_path: str):
    ffmpeg.input(input_path).output(output_path, format='mp3').run(overwrite_output=True, quiet=True)

def get_embedding(text):
    response = client.embeddings.create(input=text, model="text-embedding-3-small")
    return response.data[0].embedding

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

        # Save to temporary .mp4 file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
            temp_video.write(body)
            video_path = temp_video.name

        # Convert to mp3
        mp3_path = video_path.replace(".mp4", ".mp3")
        convert_to_mp3(video_path, mp3_path)

        # Transcribe using OpenAI Whisper
        with open(mp3_path, "rb") as audio_file:
            transcript_text = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )

        # Chunk and embed into Pinecone
        chunks = split_text(transcript_text)
        vectors = []
        for chunk in chunks:
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

# (Optional) Embedding endpoint if you want to use it separately
@app.post("/embed")
async def embed_text(request: Request):
    data = await request.json()
    embedding = get_embedding(data["text"])
    return {"embedding": embedding}

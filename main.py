from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import openai
from pinecone import Pinecone
import tempfile
import os
import uuid
import ffmpeg
import shutil

# ✅ Load API keys from environment
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# ✅ Debug check for API keys
if not openai_api_key:
    raise RuntimeError("❌ OPENAI_API_KEY not found in environment.")
print("✅ Loaded OpenAI key (first 10 chars):", openai_api_key[:10])

# ✅ Set OpenAI API key
openai.api_key = openai_api_key

# ✅ Create Pinecone client
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
    ffmpeg.input(input_path).output(output_path, format='mp3').run(overwrite_output=True)

def get_embedding(text):
    response = openai.embeddings.create(input=text, model="text-embedding-3-small")
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

        # ✅ Transcribe using OpenAI SDK v1.59.4
        with open(mp3_path, "rb") as audio_file:
            transcript_response = openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
        transcript_text = transcript_response

        # ✅ Chunk and embed into Pinecone
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

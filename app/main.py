import os
import sys

# Ensure common macOS package manager paths (Homebrew, MacPorts, etc.) are in PATH
for path in ["/opt/homebrew/bin", "/opt/homebrew/sbin", "/usr/local/bin"]:
    if os.path.exists(path) and path not in os.environ.get("PATH", ""):
        os.environ["PATH"] = f"{path}:{os.environ['PATH']}"

import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from app.audio.pipeline import run_audio_pipeline


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("audio_pipeline.api")

app = FastAPI(
    title="LUVACORD AI Core Service",
    description="Stateless audio processing pipeline for chord transcription and track isolation.",
    version="1.0.0"
)

# Set up CORS middleware for direct frontend calls if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """
    Simple health check endpoint.
    """
    return {"status": "healthy", "service": "luvacord-ai-core"}

@app.post("/process-audio", status_code=status.HTTP_200_OK)
async def process_audio(file: UploadFile = File(...)):
    """
    Accepts an MP3/WAV file, runs the AI audio processing pipeline to isolate vocal/no-vocal tracks,
    transcribes chords and score, and returns the encrypted, anonymized result.
    
    All server-side files are permanently deleted post-processing.
    """
    # Validate file extension
    filename = file.filename or "uploaded_audio.mp3"
    _, ext = filename.rsplit(".", 1) if "." in filename else ("", "")
    ext = ext.lower()
    
    if ext not in ["mp3", "wav", "mpeg"]:
        logger.warning(f"Rejected upload with unsupported extension: {ext}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Unsupported file format. Please upload a WAV or MP3 audio file."
        )

    logger.info(f"Received audio processing request for file: {filename}")
    
    try:
        # Read the entire file stream into memory
        file_bytes = await file.read()
        
        # Execute the pipeline
        result = await run_audio_pipeline(file_bytes, filename)
        
        return result
        
    except Exception as e:
        logger.error(f"Error processing audio request: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing the audio: {str(e)}"
        )

from fastapi.responses import StreamingResponse
from app.audio.transcription import transcribe_lyrics
from app.audio.pipeline import JOBS_BASE_DIR
from app.audio.cleanup import cleanup_path
import json
import asyncio
import queue
import threading

def run_generator_in_thread(gen_func, *args, **kwargs):
    q = queue.Queue()
    
    def worker():
        try:
            for item in gen_func(*args, **kwargs):
                q.put((item, None))
        except Exception as e:
            q.put((None, e))
        finally:
            q.put((None, None))
            
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return q

async def lyrics_stream_generator(job_id: str, vocal_filename: str):
    job_dir = os.path.join(JOBS_BASE_DIR, job_id)
    audio_path = os.path.join(job_dir, vocal_filename)
    
    try:
        if not os.path.exists(audio_path):
            logger.error(f"Vocal file not found for streaming: {audio_path}")
            yield f"data: {json.dumps({'error': 'Vocal file not found'})}\n\n"
            return

        logger.info(f"Starting asynchronous lyrics streaming for job: {job_id}")
        
        # Run the CPU/network-heavy transcription generator in a background thread
        q = run_generator_in_thread(transcribe_lyrics, audio_path)
        
        loop = asyncio.get_running_loop()
        while True:
            # We run q.get in the executor to avoid blocking the asyncio event loop
            item, err = await loop.run_in_executor(None, q.get)
            if err is not None:
                raise err
            if item is None:
                break
            
            yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
            
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        logger.error(f"Error in lyrics streaming generator: {e}", exc_info=True)
        yield f"data: {json.dumps({'error': str(e)})}\n\n"
    finally:
        # Clean up the session workspace once streaming is finished or aborted by client
        logger.info(f"Session complete for job {job_id}. Running cleanup on: {job_dir}")
        cleanup_path(job_dir)

@app.get("/stream-lyrics")
async def stream_lyrics(job_id: str, vocal_filename: str):
    """
    Streams transcribed lyrics segment by segment as Server-Sent Events (SSE).
    Triggers absolute session directory cleanup once stream finishes.
    """
    if not job_id or not vocal_filename or ".." in job_id or ".." in vocal_filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid job ID or vocal filename."
        )
        
    return StreamingResponse(
        lyrics_stream_generator(job_id, vocal_filename),
        media_type="text/event-stream"
    )

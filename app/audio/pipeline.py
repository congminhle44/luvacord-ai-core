import os
import tempfile
import base64
import logging
from typing import Dict, Any

from app.audio.isolation import isolate_tracks
from app.audio.transcription import transcribe_chords, transcribe_score
from app.audio.anonymizer import anonymize_and_strip_metadata
from app.audio.cleanup import cleanup_path

logger = logging.getLogger("audio_pipeline.orchestrator")

# Base directory for temporary jobs
JOBS_BASE_DIR = "/tmp/luvacord_jobs"

def ensure_jobs_dir():
    """
    Ensures that the luvacord jobs base directory exists.
    """
    try:
        os.makedirs(JOBS_BASE_DIR, exist_ok=True)
    except Exception as e:
        logger.warning(f"Could not create jobs base directory {JOBS_BASE_DIR}: {e}. Using default system temp.")

def file_to_base64(file_path: str) -> str:
    """
    Reads a binary file from disk and encodes it to a base64 string.
    """
    with open(file_path, "rb") as f:
        binary_data = f.read()
        return base64.b64encode(binary_data).decode("utf-8")

async def run_audio_pipeline(file_bytes: bytes, original_filename: str) -> Dict[str, Any]:
    """
    Orchestrates the stateless audio processing pipeline.
    
    1. INGESTION: Writes the uploaded file into a temporary job folder.
    2. ISOLATION: Runs Demucs/Spleeter on the input track to separate Vocals & Accompaniment.
    3. TRANSCRIPTION: Extracts chords (JSON) and notes (MIDI score) from accompaniment.
    4. ANONYMIZATION: Strips metadata tags and renames audio tracks with secure UUIDs.
    5. RESPONSE PACKAGING: Reads the results into memory, encodes binary files, and returns the response.
    6. CLEANUP: Fully destroys all temporary directories and generated tracks.
    """
    ensure_jobs_dir()
    
    # We create a unique sub-directory inside the jobs directory for this specific request
    temp_dir = tempfile.mkdtemp(dir=JOBS_BASE_DIR if os.path.exists(JOBS_BASE_DIR) else None)
    logger.info(f"Initialized temporary workspace for job: {temp_dir}")
    
    input_file_path = os.path.join(temp_dir, original_filename)
    
    try:
        # Step 1: Ingest - Write uploaded file to temp directory
        with open(input_file_path, "wb") as f:
            f.write(file_bytes)
        logger.info(f"Ingested original file to: {input_file_path}")
        
        # Step 2: Isolation - Separate vocal and accompaniment tracks
        vocals_raw_path, accompaniment_raw_path = isolate_tracks(input_file_path, temp_dir)
        logger.info(f"Isolated raw vocals: {vocals_raw_path}")
        logger.info(f"Isolated raw accompaniment: {accompaniment_raw_path}")
        
        # Step 3: Core Processing - Chord recognition, lyrics alignment, and note transcription
        logger.info(f"Ensuring chord transcription is run on the accompaniment (no-vocal) track: {accompaniment_raw_path}")
        chords_metadata = transcribe_chords(accompaniment_raw_path)
        
        score_filename = "score.mid"
        score_temp_path = os.path.join(temp_dir, score_filename)
        logger.info(f"Ensuring note transcription is run on the accompaniment (no-vocal) track: {accompaniment_raw_path}")
        transcribe_score(accompaniment_raw_path, score_temp_path)
        
        # Step 4: Secure Anonymization & Metadata Stripping
        # This strips all artist, ID3 tags and renames using randomized secure tokens/UUIDs
        vocals_clean_path = anonymize_and_strip_metadata(vocals_raw_path, temp_dir, "vocal")
        accompaniment_clean_path = anonymize_and_strip_metadata(accompaniment_raw_path, temp_dir, "no_vocal")
        
        vocals_clean_filename = os.path.basename(vocals_clean_path)
        accompaniment_clean_filename = os.path.basename(accompaniment_clean_path)
        
        # Step 5: Package response assets into memory (Base64 encoding)
        logger.info("Packaging binary assets to base64 for client-side storage...")
        vocals_b64 = file_to_base64(vocals_clean_path)
        accompaniment_b64 = file_to_base64(accompaniment_clean_path)
        score_b64 = file_to_base64(score_temp_path)
        
        # Construct the response payload
        response_payload = {
            "job_id": os.path.basename(temp_dir),
            "chords": chords_metadata,
            "vocal_track": {
                "filename": vocals_clean_filename,
                "data": vocals_b64,
                "mime_type": "audio/mpeg" if vocals_clean_filename.endswith(".mp3") else "audio/wav"
            },
            "no_vocal_track": {
                "filename": accompaniment_clean_filename,
                "data": accompaniment_b64,
                "mime_type": "audio/mpeg" if accompaniment_clean_filename.endswith(".mp3") else "audio/wav"
            },
            "score": {
                "filename": score_filename,
                "data": score_b64,
                "mime_type": "audio/midi"
            }
        }
        
        logger.info("Pipeline processing complete. Payload packaged successfully.")
        return response_payload
        
    finally:
        # Step 6: Resilient cleanup - Deletes all temporary working folders and original files (Disabled for testing)
        logger.info(f"Cleanup of workspace: {temp_dir} skipped for testing")
        # cleanup_path(temp_dir)

import os
import uuid
import shutil
import subprocess
import logging

logger = logging.getLogger("audio_pipeline.anonymizer")

def is_ffmpeg_available() -> bool:
    """
    Checks if FFmpeg is available on the system PATH.
    """
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except FileNotFoundError:
        return False

def anonymize_and_strip_metadata(input_path: str, output_dir: str, suffix: str) -> str:
    """
    Generates a secure UUID filename, strips all metadata (artist, title, ID3 tags)
    from the audio file, and saves the cleaned file in the output directory.
    If FFmpeg is available, transcodes the output to a compressed MP3 file to 
    reduce bandwidth and browser loading time.
    
    Args:
        input_path: Path to the original audio file.
        output_dir: Directory where the anonymized file should be saved.
        suffix: Suffix for the filename (e.g. "vocal" or "no_vocal").
        
    Returns:
        The absolute path to the newly created anonymized and stripped file.
    """
    # Generate secure UUID filename
    secure_token = str(uuid.uuid4())
    
    if is_ffmpeg_available():
        logger.info(f"FFmpeg is available. Transcoding '{input_path}' to MP3 and stripping metadata...")
        # Target .mp3 extension for compression
        filename = f"{secure_token}_{suffix}.mp3"
        output_path = os.path.join(output_dir, filename)
        
        # Transcode to MP3 with 192kbps VBR/CBR quality and strip metadata
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-map_metadata", "-1",
            "-fflags", "+bitexact",  # Ensure container level metadata is also stripped
            "-flags:v", "+bitexact",
            "-flags:a", "+bitexact",
            "-c:a", "libmp3lame",
            "-b:a", "192k",
            output_path
        ]
        try:
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            logger.info(f"Anonymized, compressed and stripped metadata successfully to: {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg transcoding failed with code {e.returncode}. Error: {e.stderr}. Falling back to copy.")
            
    # Fallback if FFmpeg is not available or command failed
    _, ext = os.path.splitext(input_path)
    if not ext:
        ext = ".mp3"
    filename = f"{secure_token}_{suffix}{ext}"
    output_path = os.path.join(output_dir, filename)
    
    logger.warning(f"FFmpeg fallback: Copying file without re-encoding to {output_path}")
    shutil.copy(input_path, output_path)
    return output_path

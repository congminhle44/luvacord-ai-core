import os
import shutil
import subprocess
import logging
from typing import Tuple

logger = logging.getLogger("audio_pipeline.isolation")

def run_demucs(input_path: str, output_dir: str) -> Tuple[str, str]:
    """
    Executes Meta's Demucs CLI to isolate the audio into Vocal and Accompaniment.
    Command: demucs -d <device> --two-stems=vocals -o <output_dir> <input_path>
    """
    import sys
    import torch
    logger.info(f"Attempting to run Meta Demucs on '{input_path}'...")
    
    input_filename = os.path.basename(input_path)
    input_base, _ = os.path.splitext(input_filename)
    
    # Configure environment with MPS fallback for Apple Silicon GPUs
    env = os.environ.copy()
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    
    # Detect best available device
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
        
    try:
        import demucs
        cmd = [
            sys.executable,
            "-m",
            "demucs",
            "-d", device,
            "--two-stems=vocals",
            "-o", output_dir,
            input_path
        ]
    except ImportError:
        cmd = [
            "demucs",
            "-d", device,
            "--two-stems=vocals",
            "-o", output_dir,
            input_path
        ]
    
    logger.info(f"Running Demucs command on {device}: {' '.join(cmd)}")
    # Run demucs command
    subprocess.run(cmd, env=env, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Locate the files
    demucs_model = "htdemucs"  # Default demucs model name
    vocals_path = os.path.join(output_dir, demucs_model, input_base, "vocals.wav")
    no_vocals_path = os.path.join(output_dir, demucs_model, input_base, "no_vocals.wav")
    
    # If the default model output isn't there, check other possible demucs folders
    if not os.path.exists(vocals_path):
        # Scan output_dir recursively for vocals.wav/vocals.mp3 and no_vocals.wav/no_vocals.mp3
        for root, _, files in os.walk(output_dir):
            for file in files:
                if file.startswith("vocals"):
                    vocals_path = os.path.join(root, file)
                elif file.startswith("no_vocals"):
                    no_vocals_path = os.path.join(root, file)
                    
    if not os.path.exists(vocals_path) or not os.path.exists(no_vocals_path):
        raise FileNotFoundError("Demucs completed but output files were not found.")
        
    return vocals_path, no_vocals_path

def run_spleeter(input_path: str, output_dir: str) -> Tuple[str, str]:
    """
    Executes Deezer's Spleeter CLI to isolate audio into Vocal and Accompaniment.
    Command: spleeter separate -p spleeter:2stems -o <output_dir> <input_path>
    """
    logger.info(f"Attempting to run Spleeter on '{input_path}'...")
    input_filename = os.path.basename(input_path)
    input_base, _ = os.path.splitext(input_filename)
    
    cmd = [
        "spleeter", "separate",
        "-p", "spleeter:2stems",
        "-o", output_dir,
        input_path
    ]
    
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Spleeter outputs: <output_dir>/<input_base>/vocals.wav and accompaniment.wav
    vocals_path = os.path.join(output_dir, input_base, "vocals.wav")
    no_vocals_path = os.path.join(output_dir, input_base, "accompaniment.wav")
    
    if not os.path.exists(vocals_path) or not os.path.exists(no_vocals_path):
        raise FileNotFoundError("Spleeter completed but output files were not found.")
        
    return vocals_path, no_vocals_path

def isolate_tracks(input_path: str, output_dir: str) -> Tuple[str, str]:
    """
    Tries to isolate audio using Demucs, falling back to Spleeter if Demucs is not found.
    If neither is available, falls back to a Mock isolator for testing/local dev.
    
    Returns:
        A tuple of (vocals_path, no_vocals_path)
    """
    # 1. Try Demucs
    try:
        has_demucs = False
        try:
            import demucs
            has_demucs = True
        except ImportError:
            if shutil.which("demucs"):
                has_demucs = True
                
        if has_demucs:
            return run_demucs(input_path, output_dir)
        else:
            raise FileNotFoundError("Demucs is not installed (neither importable nor CLI executable).")
    except Exception as e:
        logger.warning(f"Demucs isolation failed or not found: {e}. Trying Spleeter...")
        
    # 2. Try Spleeter
    try:
        shutil.which("spleeter")
        return run_spleeter(input_path, output_dir)
    except (subprocess.SubprocessError, FileNotFoundError, Exception) as e:
        logger.warning(f"Spleeter isolation failed or not found: {e}. Falling back to MOCK isolation.")
        
    # 3. Fallback Mock Isolation (for local testing/dev)
    logger.info("Mock Isolation: Copying input file to mock vocal and accompaniment tracks.")
    _, ext = os.path.splitext(input_path)
    if not ext:
        ext = ".wav"
    mock_vocals_path = os.path.join(output_dir, f"mock_vocals{ext}")
    mock_accompaniment_path = os.path.join(output_dir, f"mock_accompaniment{ext}")
    
    shutil.copy(input_path, mock_vocals_path)
    shutil.copy(input_path, mock_accompaniment_path)
    
    return mock_vocals_path, mock_accompaniment_path

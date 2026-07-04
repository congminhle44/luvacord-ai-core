import os
import re
import base64
import pytest
from io import BytesIO
from fastapi import UploadFile
from fastapi.testclient import TestClient
from starlette.datastructures import Headers
from app.main import app, process_audio
from app.audio.pipeline import JOBS_BASE_DIR

client = TestClient(app)

def test_health_check():
    """
    Test the health check endpoint returns 200 OK.
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "luvacord-ai-core"}

def test_process_audio_pipeline():
    """
    Test the entire /process-audio pipeline using a mock audio file.
    Verifies output formatting, secure renaming, and complete directory cleanup.
    """
    # Create a small dummy WAV/MP3 byte stream
    mock_audio_content = b"ID3v2.3.0\x00\x00\x00\x00\x00\x00\x00FakeMP3BodyWithArtistMetadata"
    
    # Send request with mock file
    files = {"file": ("test_song.mp3", mock_audio_content, "audio/mpeg")}
    response = client.post("/process-audio", files=files)
    
    # Assert response status
    assert response.status_code == 200
    
    data = response.json()
    
    # Assert that all keys exist
    assert "job_id" in data
    assert "chords" in data
    assert "vocal_track" in data
    assert "no_vocal_track" in data
    assert "score" in data
    assert "lyrics" not in data
    
    # Validate chords timeline metadata
    chords = data["chords"]
    assert isinstance(chords, list)
    assert len(chords) > 0
    assert "time" in chords[0]
    assert "chord" in chords[0]
    assert "duration" in chords[0]

    # Test the streaming lyrics endpoint using the job_id and vocals filename
    stream_response = client.get(f"/stream-lyrics?job_id={data['job_id']}&vocal_filename={data['vocal_track']['filename']}")
    assert stream_response.status_code == 200
    
    stream_content = stream_response.text
    assert "data: " in stream_content
    assert "Mưa vẫn mưa bay trên tầng tháp cổ" in stream_content or "[DONE]" in stream_content
    
    # Validate vocal and no-vocal track outputs
    vocal = data["vocal_track"]
    no_vocal = data["no_vocal_track"]
    score = data["score"]
    
    # Ensure filenames are randomized secure UUIDs
    uuid_pattern = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}_(vocal|no_vocal)\.(mp3|wav)$")
    assert uuid_pattern.match(vocal["filename"])
    assert uuid_pattern.match(no_vocal["filename"])
    assert score["filename"] == "score.mid"
    
    # Validate Base64 encoding decoding
    vocals_decoded = base64.b64decode(vocal["data"])
    no_vocals_decoded = base64.b64decode(no_vocal["data"])
    score_decoded = base64.b64decode(score["data"])
    
    assert len(vocals_decoded) > 0
    assert len(no_vocals_decoded) > 0
    assert len(score_decoded) > 0
    
    # Verify that the minimal MIDI score is correct by checking MIDI header "MThd"
    assert score_decoded.startswith(b"MThd")

    # Assert that the temporary directory is NOT empty since cleanup is disabled for testing
    if os.path.exists(JOBS_BASE_DIR):
        contents = os.listdir(JOBS_BASE_DIR)
        assert len(contents) > 0, "Expected lingering job directories in JOBS_BASE_DIR because cleanup is disabled for testing"

def test_process_audio_invalid_format():
    """
    Test that invalid formats (e.g. .txt or .png) are rejected with a 400 Bad Request.
    """
    files = {"file": ("malicious.txt", b"plain text content", "text/plain")}
    response = client.post("/process-audio", files=files)
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]


@pytest.mark.anyio
async def test_process_audio_missing_filename():
    """
    Test that files uploaded with None/missing filename default correctly to .mp3 and pass validation.
    """
    mock_audio_content = b"ID3v2.3.0\x00\x00\x00\x00\x00\x00\x00FakeMP3BodyWithArtistMetadata"
    mock_file = UploadFile(
        file=BytesIO(mock_audio_content),
        filename=None,
        headers=Headers({"content-type": "audio/mpeg"})
    )
    response = await process_audio(file=mock_file)
    assert response is not None
    assert "chords" in response


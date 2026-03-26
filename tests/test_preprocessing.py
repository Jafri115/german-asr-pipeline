"""Tests for preprocessing module."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from asr_pipeline.preprocessing import (
    AudioChunker,
    AudioPreprocessor,
    TranscriptNormalizer,
)


def create_test_audio(path: Path, duration: float = 1.0, sr: int = 16000):
    """Create a test audio file."""
    samples = int(duration * sr)
    audio = np.random.randn(samples) * 0.1
    sf.write(path, audio, sr)


def test_transcript_normalizer():
    """Test transcript normalization."""
    normalizer = TranscriptNormalizer(
        lowercase=True,
        remove_punctuation=True,
        normalize_numbers=False,
        remove_filler_words=False,
        remove_annotations=True,
    )
    
    # Test lowercase
    assert normalizer.normalize("HELLO WORLD") == "hello world"
    
    # Test annotation removal
    assert normalizer.normalize("hello [noise] world") == "hello world"
    assert normalizer.normalize("hello (laughs) world") == "hello world"
    
    # Test punctuation removal
    assert normalizer.normalize("hello, world!") == "hello world"
    
    # Test whitespace normalization
    assert normalizer.normalize("hello    world") == "hello world"


def test_transcript_normalizer_german():
    """Test German-specific normalization."""
    normalizer = TranscriptNormalizer(lowercase=True)
    
    # Test German text
    text = "Hallo, wie geht es Ihnen?"
    result = normalizer.normalize(text)
    assert result == "hallo wie geht es ihnen"


def test_audio_preprocessor():
    """Test audio preprocessing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test audio
        input_path = Path(tmpdir) / "input.wav"
        output_path = Path(tmpdir) / "output.wav"
        create_test_audio(input_path, duration=2.0, sr=16000)
        
        preprocessor = AudioPreprocessor(
            target_sample_rate=16000,
            normalize_volume=True,
            trim_silence=False,
        )
        
        waveform, sr, saved_path = preprocessor.process_audio(
            input_path, output_path
        )
        
        assert saved_path == output_path
        assert saved_path.exists()
        assert sr == 16000
        assert len(waveform) == 2 * 16000  # 2 seconds at 16kHz


def test_audio_chunker():
    """Test audio chunking."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create long test audio
        input_path = Path(tmpdir) / "input.wav"
        create_test_audio(input_path, duration=60.0, sr=16000)
        
        chunker = AudioChunker(
            chunk_duration_sec=30.0,
            overlap_sec=1.0,
            min_chunk_duration_sec=5.0,
        )
        
        # Load and chunk
        import librosa
        waveform, sr = librosa.load(input_path, sr=16000)
        chunks = chunker.chunk_audio(waveform, sr)
        
        # Should get 2 chunks for 60s audio with 30s chunks
        assert len(chunks) >= 1
        
        # Each chunk should have correct structure
        for chunk, start_sec, end_sec in chunks:
            assert isinstance(chunk, np.ndarray)
            assert isinstance(start_sec, float)
            assert isinstance(end_sec, float)
            assert end_sec > start_sec


def test_audio_chunker_manifest():
    """Test manifest-based chunking."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test audio and manifest
        audio_dir = Path(tmpdir) / "audio"
        audio_dir.mkdir()
        
        audio_path = audio_dir / "test.wav"
        create_test_audio(audio_path, duration=45.0, sr=16000)
        
        manifest_path = Path(tmpdir) / "manifest.parquet"
        df = pd.DataFrame({
            "sample_id": ["s1"],
            "audio_path": [str(audio_path)],
            "duration_sec": [45.0],
            "is_valid": [True],
        })
        df.to_parquet(manifest_path)
        
        output_dir = Path(tmpdir) / "chunks"
        output_manifest = Path(tmpdir) / "chunked_manifest.parquet"
        
        chunker = AudioChunker(
            chunk_duration_sec=30.0,
            overlap_sec=1.0,
            min_chunk_duration_sec=5.0,
        )
        
        result_df = chunker.chunk_manifest(
            manifest_path=manifest_path,
            output_dir=output_dir,
            output_manifest_path=output_manifest,
            max_duration_threshold=30.0,
        )
        
        # Should have created chunks
        assert len(result_df) >= 1
        assert output_manifest.exists()

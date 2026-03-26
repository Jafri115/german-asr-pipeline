"""Tests for model wrappers."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from asr_pipeline.models import (
    ASRModel,
    ModelRegistry,
    TranscriptNormalizer,
)


def create_test_audio(path: Path, duration: float = 1.0, sr: int = 16000):
    """Create a test audio file."""
    samples = int(duration * sr)
    audio = np.random.randn(samples) * 0.1
    sf.write(path, audio, sr)


def test_model_registry_list_models():
    """Test listing available models."""
    models_df = ModelRegistry.list_models()
    
    assert len(models_df) > 0
    assert "key" in models_df.columns
    assert "model_name" in models_df.columns
    assert "description" in models_df.columns
    
    # Check that expected models are present
    keys = models_df["key"].tolist()
    assert "whisper-large-v3-turbo" in keys
    assert "wav2vec2-xlsr-german" in keys


def test_model_registry_get_model():
    """Test getting a model from registry."""
    # Skip actual model loading in tests
    with patch("asr_pipeline.models.WhisperModel.load") as mock_load:
        model = ModelRegistry.get_model("whisper-large-v3-turbo")
        
        assert model is not None
        assert model.model_name == "openai/whisper-large-v3-turbo"
        assert model.language == "german"


def test_model_registry_unknown_model():
    """Test getting an unknown model raises error."""
    with pytest.raises(ValueError):
        ModelRegistry.get_model("unknown-model")


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


@pytest.mark.skip(reason="Requires actual model download")
def test_whisper_model_transcribe():
    """Test Whisper transcription (requires model)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = Path(tmpdir) / "test.wav"
        create_test_audio(audio_path, duration=2.0)
        
        from asr_pipeline.models import WhisperModel
        
        model = WhisperModel("openai/whisper-tiny")  # Use tiny for speed
        model.load()
        
        result = model.transcribe(audio_path)
        
        assert "transcription" in result
        assert "inference_time_sec" in result
        assert isinstance(result["transcription"], str)


@pytest.mark.skip(reason="Requires actual model download")
def test_wav2vec2_model_transcribe():
    """Test Wav2Vec2 transcription (requires model)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = Path(tmpdir) / "test.wav"
        create_test_audio(audio_path, duration=2.0)
        
        from asr_pipeline.models import Wav2Vec2Model
        
        model = Wav2Vec2Model("facebook/wav2vec2-base-960h")
        model.load()
        
        result = model.transcribe(audio_path)
        
        assert "transcription" in result
        assert "inference_time_sec" in result

"""Model wrappers for ASR models."""

import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torchaudio
from transformers import (
    AutoModelForCTC,
    AutoProcessor,
    AutoTokenizer,
    pipeline,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
from tqdm import tqdm

from asr_pipeline.utils import get_logger

logger = get_logger(__name__)


class ASRModel(ABC):
    """Abstract base class for ASR models."""
    
    def __init__(self, model_name: str, device: Optional[str] = None):
        """Initialize model.
        
        Args:
            model_name: Model identifier
            device: Device to use (cuda/cpu)
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self.tokenizer = None
        
    @abstractmethod
    def load(self) -> None:
        """Load the model."""
        pass
    
    @abstractmethod
    def transcribe(
        self,
        audio_path: Union[str, Path],
    ) -> Dict[str, Union[str, float]]:
        """Transcribe a single audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dict with transcription and metadata
        """
        pass
    
    def transcribe_batch(
        self,
        audio_paths: List[Union[str, Path]],
        batch_size: int = 8,
    ) -> List[Dict[str, Union[str, float]]]:
        """Transcribe multiple audio files.
        
        Args:
            audio_paths: List of audio paths
            batch_size: Batch size
            
        Returns:
            List of transcription results
        """
        results = []
        for i in tqdm(range(0, len(audio_paths), batch_size), desc="Transcribing"):
            batch = audio_paths[i:i + batch_size]
            for audio_path in batch:
                result = self.transcribe(audio_path)
                results.append(result)
        return results
    
    def unload(self) -> None:
        """Unload model to free memory."""
        self.model = None
        self.processor = None
        self.tokenizer = None
        torch.cuda.empty_cache()


class WhisperModel(ASRModel):
    """Wrapper for OpenAI Whisper models."""
    
    def __init__(
        self,
        model_name: str = "openai/whisper-large-v3",
        device: Optional[str] = None,
        language: str = "german",
        task: str = "transcribe",
        torch_dtype: torch.dtype = torch.float16,
    ):
        """Initialize Whisper model.
        
        Args:
            model_name: HuggingFace model name
            device: Device to use
            language: Language code
            task: Task (transcribe/translate)
            torch_dtype: Torch dtype for inference
        """
        super().__init__(model_name, device)
        self.language = language
        self.task = task
        self.torch_dtype = torch_dtype
        
    def load(self) -> None:
        """Load Whisper model and processor."""
        logger.info(f"Loading Whisper model: {self.model_name}")
        
        self.processor = WhisperProcessor.from_pretrained(self.model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
            device_map=self.device if self.device != "cpu" else None,
        )
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        
        # Set language and task
        self.model.config.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=self.language,
            task=self.task,
        )
        
        self.model.eval()
        logger.info(f"Model loaded on {self.device}")
    
    def transcribe(
        self,
        audio_path: Union[str, Path],
    ) -> Dict[str, Union[str, float]]:
        """Transcribe audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dict with transcription and metadata
        """
        if self.model is None:
            self.load()
        
        start_time = time.time()
        
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Process
        input_features = self.processor(
            waveform.squeeze().numpy(),
            sampling_rate=16000,
            return_tensors="pt",
        ).input_features
        
        input_features = input_features.to(self.device)
        
        # Generate
        with torch.no_grad():
            predicted_ids = self.model.generate(input_features)
        
        transcription = self.processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True,
        )[0]
        
        inference_time = time.time() - start_time
        
        return {
            "transcription": transcription,
            "inference_time_sec": inference_time,
            "model_name": self.model_name,
        }


class Wav2Vec2Model(ASRModel):
    """Wrapper for Wav2Vec2/XLS-R models."""
    
    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-large-xlsr-53-german",
        device: Optional[str] = None,
        torch_dtype: torch.dtype = torch.float32,
    ):
        """Initialize Wav2Vec2 model.
        
        Args:
            model_name: HuggingFace model name
            device: Device to use
            torch_dtype: Torch dtype for inference
        """
        super().__init__(model_name, device)
        self.torch_dtype = torch_dtype
    
    def load(self) -> None:
        """Load Wav2Vec2 model and processor."""
        logger.info(f"Loading Wav2Vec2 model: {self.model_name}")
        
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = AutoModelForCTC.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
        )
        
        self.model = self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded on {self.device}")
    
    def transcribe(
        self,
        audio_path: Union[str, Path],
    ) -> Dict[str, Union[str, float]]:
        """Transcribe audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dict with transcription and metadata
        """
        if self.model is None:
            self.load()
        
        start_time = time.time()
        
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Process
        inputs = self.processor(
            waveform.squeeze().numpy(),
            sampling_rate=16000,
            return_tensors="pt",
        )
        
        input_values = inputs.input_values.to(self.device)
        
        # Generate
        with torch.no_grad():
            logits = self.model(input_values).logits
        
        # Decode
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]
        
        inference_time = time.time() - start_time
        
        return {
            "transcription": transcription,
            "inference_time_sec": inference_time,
            "model_name": self.model_name,
        }


class HuggingFacePipelineModel(ASRModel):
    """Wrapper using HuggingFace pipeline for ASR."""
    
    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        chunk_length_s: float = 30.0,
        stride_length_s: float = 5.0,
        language: str = "german",
    ):
        """Initialize pipeline model.
        
        Args:
            model_name: HuggingFace model name
            device: Device to use
            chunk_length_s: Chunk length in seconds
            stride_length_s: Stride length in seconds
            language: Language code
        """
        super().__init__(model_name, device)
        self.chunk_length_s = chunk_length_s
        self.stride_length_s = stride_length_s
        self.language = language
        self.pipe = None
    
    def load(self) -> None:
        """Load pipeline."""
        logger.info(f"Loading pipeline model: {self.model_name}")
        
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model_name,
            device=0 if self.device == "cuda" else -1,
            chunk_length_s=self.chunk_length_s,
            stride_length_s=self.stride_length_s,
        )
        
        logger.info("Pipeline loaded")
    
    def transcribe(
        self,
        audio_path: Union[str, Path],
    ) -> Dict[str, Union[str, float]]:
        """Transcribe audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dict with transcription and metadata
        """
        if self.pipe is None:
            self.load()
        
        start_time = time.time()
        
        result = self.pipe(
            str(audio_path),
            generate_kwargs={"language": self.language},
        )
        
        inference_time = time.time() - start_time
        
        return {
            "transcription": result["text"],
            "inference_time_sec": inference_time,
            "model_name": self.model_name,
        }


class ModelRegistry:
    """Registry of available ASR models."""
    
    # Recommended models for German ASR
    GERMAN_MODELS = {
        # Primary recommendation: Whisper Large V3 Turbo
        "whisper-large-v3-turbo": {
            "class": WhisperModel,
            "name": "openai/whisper-large-v3-turbo",
            "description": "Fast, accurate multilingual ASR (recommended primary)",
            "language": "german",
        },
        # Fallback: Whisper Large V3
        "whisper-large-v3": {
            "class": WhisperModel,
            "name": "openai/whisper-large-v3",
            "description": "Most accurate Whisper model",
            "language": "german",
        },
        # Fallback: Whisper Medium
        "whisper-medium": {
            "class": WhisperModel,
            "name": "openai/whisper-medium",
            "description": "Medium-sized Whisper model",
            "language": "german",
        },
        # Wav2Vec2 German
        "wav2vec2-xlsr-german": {
            "class": Wav2Vec2Model,
            "name": "facebook/wav2vec2-large-xlsr-53-german",
            "description": "German fine-tuned XLS-R (recommended fallback)",
        },
        # Nvidia Canary (if available)
        "canary-1b": {
            "class": HuggingFacePipelineModel,
            "name": "nvidia/canary-1b",
            "description": "Nvidia Canary 1B multilingual",
            "language": "de",
        },
    }
    
    @classmethod
    def get_model(cls, model_key: str, device: Optional[str] = None) -> ASRModel:
        """Get a model by key.
        
        Args:
            model_key: Key in GERMAN_MODELS
            device: Device to use
            
        Returns:
            ASRModel instance
        """
        if model_key not in cls.GERMAN_MODELS:
            raise ValueError(f"Unknown model: {model_key}. Available: {list(cls.GERMAN_MODELS.keys())}")
        
        config = cls.GERMAN_MODELS[model_key]
        model_class = config["class"]
        
        kwargs = {"device": device}
        if "language" in config:
            kwargs["language"] = config["language"]
        
        return model_class(model_name=config["name"], **kwargs)
    
    @classmethod
    def list_models(cls) -> pd.DataFrame:
        """List all available models.
        
        Returns:
            DataFrame with model info
        """
        data = []
        for key, config in cls.GERMAN_MODELS.items():
            data.append({
                "key": key,
                "model_name": config["name"],
                "description": config["description"],
                "class": config["class"].__name__,
            })
        return pd.DataFrame(data)


def load_model(
    model_name_or_path: str,
    model_type: str = "auto",
    device: Optional[str] = None,
) -> ASRModel:
    """Load an ASR model.
    
    Args:
        model_name_or_path: Model name or path
        model_type: Model type (whisper, wav2vec2, auto)
        device: Device to use
        
    Returns:
        ASRModel instance
    """
    # Check if it's a registry key
    if model_name_or_path in ModelRegistry.GERMAN_MODELS:
        return ModelRegistry.get_model(model_name_or_path, device)
    
    # Auto-detect type
    if model_type == "auto":
        if "whisper" in model_name_or_path.lower():
            model_type = "whisper"
        elif "wav2vec" in model_name_or_path.lower() or "xlsr" in model_name_or_path.lower():
            model_type = "wav2vec2"
        else:
            model_type = "pipeline"
    
    # Create model
    if model_type == "whisper":
        return WhisperModel(model_name_or_path, device)
    elif model_type == "wav2vec2":
        return Wav2Vec2Model(model_name_or_path, device)
    else:
        return HuggingFacePipelineModel(model_name_or_path, device)

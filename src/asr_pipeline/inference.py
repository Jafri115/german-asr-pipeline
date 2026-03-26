"""Batch inference module for running transcription on full datasets."""

from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
from tqdm import tqdm

from asr_pipeline.models import load_model
from asr_pipeline.utils import get_logger, load_manifest, save_manifest

logger = get_logger(__name__)


class BatchInference:
    """Run batch inference on audio files."""
    
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        batch_size: int = 1,
        use_fallback: bool = False,
        fallback_model_path: Optional[str] = None,
    ):
        """Initialize batch inference.
        
        Args:
            model_path: Path to primary model
            device: Device to use
            batch_size: Batch size (currently only 1 supported for most models)
            use_fallback: Whether to use fallback on failures
            fallback_model_path: Path to fallback model
        """
        self.model_path = model_path
        self.device = device
        self.batch_size = batch_size
        self.use_fallback = use_fallback
        self.fallback_model_path = fallback_model_path
        
        self.primary_model = None
        self.fallback_model = None
    
    def load_models(self) -> None:
        """Load primary and fallback models."""
        logger.info(f"Loading primary model: {self.model_path}")
        self.primary_model = load_model(self.model_path, device=self.device)
        self.primary_model.load()
        
        if self.use_fallback and self.fallback_model_path:
            logger.info(f"Loading fallback model: {self.fallback_model_path}")
            self.fallback_model = load_model(self.fallback_model_path, device=self.device)
            self.fallback_model.load()
    
    def transcribe_with_fallback(
        self,
        audio_path: Union[str, Path],
    ) -> Dict:
        """Transcribe with fallback on failure.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcription result
        """
        # Try primary model
        try:
            result = self.primary_model.transcribe(audio_path)
            result["model_used"] = "primary"
            return result
        except Exception as e:
            logger.warning(f"Primary model failed for {audio_path}: {e}")
            
            if self.use_fallback and self.fallback_model:
                try:
                    result = self.fallback_model.transcribe(audio_path)
                    result["model_used"] = "fallback"
                    result["primary_error"] = str(e)
                    return result
                except Exception as fallback_e:
                    logger.error(f"Fallback model also failed: {fallback_e}")
                    return {
                        "transcription": "",
                        "inference_time_sec": 0,
                        "error": f"Primary: {e}, Fallback: {fallback_e}",
                        "model_used": "none",
                    }
            else:
                return {
                    "transcription": "",
                    "inference_time_sec": 0,
                    "error": str(e),
                    "model_used": "none",
                }
    
    def run_inference(
        self,
        manifest_path: Union[str, Path],
        output_path: Union[str, Path],
        transcript_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """Run inference on all samples in manifest.
        
        Args:
            manifest_path: Path to input manifest
            output_path: Path to save output manifest
            transcript_col: Optional column with existing transcripts to compare
            
        Returns:
            Output DataFrame with transcriptions
        """
        if self.primary_model is None:
            self.load_models()
        
        df = load_manifest(manifest_path)
        
        # Filter to valid samples
        if "is_valid" in df.columns:
            df = df[df["is_valid"]].copy()
        
        logger.info(f"Running inference on {len(df)} samples")
        
        results = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Transcribing"):
            audio_path = row["audio_path"]
            
            result = self.transcribe_with_fallback(audio_path)
            
            record = row.to_dict()
            record["predicted_transcript"] = result.get("transcription", "")
            record["inference_time_sec"] = result.get("inference_time_sec", 0)
            record["model_used"] = result.get("model_used", "unknown")
            
            if "error" in result:
                record["inference_error"] = result["error"]
            
            results.append(record)
        
        results_df = pd.DataFrame(results)
        
        # Compute metrics if reference transcripts available
        if transcript_col and transcript_col in results_df.columns:
            import jiwer
            
            valid_pairs = results_df[
                (results_df[transcript_col].notna()) &
                (results_df["predicted_transcript"] != "")
            ]
            
            if len(valid_pairs) > 0:
                wer = jiwer.wer(
                    valid_pairs[transcript_col].tolist(),
                    valid_pairs["predicted_transcript"].tolist(),
                ) * 100
                
                results_df["dataset_wer"] = wer
                logger.info(f"Dataset WER: {wer:.2f}%")
        
        # Save results
        save_manifest(results_df, output_path)
        
        # Print summary
        primary_count = (results_df["model_used"] == "primary").sum()
        fallback_count = (results_df["model_used"] == "fallback").sum()
        error_count = (results_df["model_used"] == "none").sum()
        
        logger.info(f"Inference complete:")
        logger.info(f"  Primary model: {primary_count} samples")
        logger.info(f"  Fallback model: {fallback_count} samples")
        logger.info(f"  Failed: {error_count} samples")
        
        return results_df
    
    def run_inference_on_splits(
        self,
        splits_dir: Union[str, Path],
        output_dir: Union[str, Path],
        splits: List[str] = ["train", "val", "test"],
    ) -> Dict[str, pd.DataFrame]:
        """Run inference on multiple splits.
        
        Args:
            splits_dir: Directory containing split manifests
            output_dir: Directory for output manifests
            splits: List of split names
            
        Returns:
            Dict of split name to DataFrame
        """
        splits_dir = Path(splits_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        for split in splits:
            manifest_path = splits_dir / f"{split}.parquet"
            
            if not manifest_path.exists():
                logger.warning(f"Split manifest not found: {manifest_path}")
                continue
            
            output_path = output_dir / f"{split}_predictions.parquet"
            
            logger.info(f"Running inference on {split} split")
            df = self.run_inference(manifest_path, output_path)
            results[split] = df
        
        return results


class StreamingInference:
    """Streaming inference for real-time or large-scale processing."""
    
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        chunk_length_s: float = 30.0,
    ):
        """Initialize streaming inference.
        
        Args:
            model_path: Path to model
            device: Device to use
            chunk_length_s: Chunk length in seconds
        """
        self.model_path = model_path
        self.device = device
        self.chunk_length_s = chunk_length_s
        self.model = None
    
    def load(self) -> None:
        """Load model."""
        from transformers import pipeline
        
        logger.info(f"Loading streaming model: {self.model_path}")
        
        self.model = pipeline(
            "automatic-speech-recognition",
            model=self.model_path,
            device=0 if self.device == "cuda" else -1,
            chunk_length_s=self.chunk_length_s,
        )
        
        logger.info("Model loaded")
    
    def transcribe_streaming(
        self,
        audio_path: Union[str, Path],
    ) -> Dict:
        """Transcribe with streaming.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcription result
        """
        if self.model is None:
            self.load()
        
        import time
        start_time = time.time()
        
        result = self.model(str(audio_path), return_timestamps=True)
        
        inference_time = time.time() - start_time
        
        return {
            "transcription": result["text"],
            "chunks": result.get("chunks", []),
            "inference_time_sec": inference_time,
        }


def run_full_inference(
    model_path: Union[str, Path],
    manifest_path: Union[str, Path],
    output_path: Union[str, Path],
    fallback_model_path: Optional[Union[str, Path]] = None,
    device: Optional[str] = None,
) -> pd.DataFrame:
    """Run full inference from CLI.
    
    Args:
        model_path: Path to primary model
        manifest_path: Path to input manifest
        output_path: Path to save output
        fallback_model_path: Optional fallback model path
        device: Device to use
        
    Returns:
        Results DataFrame
    """
    inference = BatchInference(
        model_path=str(model_path),
        device=device,
        use_fallback=fallback_model_path is not None,
        fallback_model_path=str(fallback_model_path) if fallback_model_path else None,
    )
    
    return inference.run_inference(manifest_path, output_path)

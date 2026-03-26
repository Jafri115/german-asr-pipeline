"""Model benchmarking module for comparing ASR models."""

import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import jiwer
import numpy as np
import pandas as pd
from tqdm import tqdm

from asr_pipeline.models import ModelRegistry, load_model
from asr_pipeline.utils import get_logger, load_manifest, save_manifest

logger = get_logger(__name__)


class MetricsComputer:
    """Compute ASR metrics."""
    
    @staticmethod
    def compute_wer(
        references: List[str],
        predictions: List[str],
    ) -> Dict[str, float]:
        """Compute Word Error Rate.
        
        Args:
            references: List of reference transcripts
            predictions: List of predicted transcripts
            
        Returns:
            Dict with WER metrics
        """
        wer = jiwer.wer(references, predictions)
        mer = jiwer.mer(references, predictions)
        wil = jiwer.wil(references, predictions)
        
        return {
            "wer": wer * 100,  # Convert to percentage
            "mer": mer * 100,
            "wil": wil * 100,
        }
    
    @staticmethod
    def compute_cer(
        references: List[str],
        predictions: List[str],
    ) -> float:
        """Compute Character Error Rate.
        
        Args:
            references: List of reference transcripts
            predictions: List of predicted transcripts
            
        Returns:
            CER as percentage
        """
        cer = jiwer.cer(references, predictions)
        return cer * 100
    
    @staticmethod
    def compute_all_metrics(
        references: List[str],
        predictions: List[str],
    ) -> Dict[str, float]:
        """Compute all metrics.
        
        Args:
            references: List of reference transcripts
            predictions: List of predicted transcripts
            
        Returns:
            Dict with all metrics
        """
        wer_metrics = MetricsComputer.compute_wer(references, predictions)
        cer = MetricsComputer.compute_cer(references, predictions)
        
        return {
            **wer_metrics,
            "cer": cer,
        }
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text for fair comparison.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        import re
        
        # Lowercase
        text = text.lower()
        
        # Remove punctuation
        text = re.sub(r"[^\w\s]", "", text)
        
        # Normalize whitespace
        text = " ".join(text.split())
        
        return text.strip()
    
    @classmethod
    def compute_normalized_metrics(
        cls,
        references: List[str],
        predictions: List[str],
    ) -> Dict[str, float]:
        """Compute metrics on normalized text.
        
        Args:
            references: List of reference transcripts
            predictions: List of predicted transcripts
            
        Returns:
            Dict with normalized metrics
        """
        norm_refs = [cls.normalize_text(r) for r in references]
        norm_preds = [cls.normalize_text(p) for p in predictions]
        
        metrics = cls.compute_all_metrics(norm_refs, norm_preds)
        
        # Rename keys
        return {f"normalized_{k}": v for k, v in metrics.items()}


class ModelBenchmark:
    """Benchmark ASR models."""
    
    def __init__(
        self,
        models: List[str],
        device: Optional[str] = None,
        use_normalized_metrics: bool = True,
    ):
        """Initialize benchmark.
        
        Args:
            models: List of model keys or names
            device: Device to use
            use_normalized_metrics: Whether to compute normalized metrics
        """
        self.models = models
        self.device = device
        self.use_normalized_metrics = use_normalized_metrics
        self.results = []
    
    def run_benchmark(
        self,
        manifest_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        max_samples: Optional[int] = None,
        transcript_col: str = "transcript_normalized",
    ) -> pd.DataFrame:
        """Run benchmark on all models.
        
        Args:
            manifest_path: Path to test manifest
            output_path: Optional path to save results
            max_samples: Maximum samples to test (for quick tests)
            transcript_col: Column with reference transcripts
            
        Returns:
            DataFrame with benchmark results
        """
        df = load_manifest(manifest_path)
        
        # Filter to valid samples with transcripts
        if "is_valid" in df.columns:
            df = df[df["is_valid"]]
        
        if transcript_col not in df.columns:
            # Fall back to raw transcript
            transcript_col = "transcript_raw"
        
        df = df[df[transcript_col].notna()]
        
        if max_samples:
            df = df.sample(min(max_samples, len(df)), random_state=42)
        
        logger.info(f"Benchmarking on {len(df)} samples")
        
        all_results = []
        
        for model_key in self.models:
            logger.info(f"Benchmarking model: {model_key}")
            
            try:
                model_results = self._benchmark_model(
                    model_key,
                    df,
                    transcript_col,
                )
                all_results.extend(model_results)
            except Exception as e:
                logger.error(f"Failed to benchmark {model_key}: {e}")
        
        results_df = pd.DataFrame(all_results)
        
        if output_path:
            save_manifest(results_df, output_path)
        
        return results_df
    
    def _benchmark_model(
        self,
        model_key: str,
        df: pd.DataFrame,
        transcript_col: str,
    ) -> List[Dict]:
        """Benchmark a single model.
        
        Args:
            model_key: Model key
            df: Test DataFrame
            transcript_col: Transcript column
            
        Returns:
            List of result dicts
        """
        # Load model
        model = load_model(model_key, device=self.device)
        model.load()
        
        results = []
        predictions = []
        references = []
        inference_times = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Running {model_key}"):
            audio_path = row["audio_path"]
            reference = row[transcript_col]
            
            try:
                result = model.transcribe(audio_path)
                prediction = result["transcription"]
                inference_time = result["inference_time_sec"]
                
                predictions.append(prediction)
                references.append(reference)
                inference_times.append(inference_time)
                
                results.append({
                    "model": model_key,
                    "sample_id": row.get("sample_id", idx),
                    "reference": reference,
                    "prediction": prediction,
                    "inference_time_sec": inference_time,
                })
                
            except Exception as e:
                logger.warning(f"Failed to transcribe {audio_path}: {e}")
                results.append({
                    "model": model_key,
                    "sample_id": row.get("sample_id", idx),
                    "reference": reference,
                    "prediction": "",
                    "inference_time_sec": 0,
                    "error": str(e),
                })
        
        model.unload()
        
        # Compute metrics
        if predictions:
            metrics = MetricsComputer.compute_all_metrics(references, predictions)
            
            # Add speed metrics
            total_duration = df["duration_sec"].sum() if "duration_sec" in df.columns else 0
            total_inference_time = sum(inference_times)
            
            speed_metrics = {
                "avg_inference_time_sec": np.mean(inference_times),
                "total_inference_time_sec": total_inference_time,
                "rtf": total_inference_time / total_duration if total_duration > 0 else 0,
                "samples_per_second": len(predictions) / total_inference_time if total_inference_time > 0 else 0,
            }
            
            # Add metrics to all results
            for result in results:
                result.update(metrics)
                result.update(speed_metrics)
            
            logger.info(f"{model_key} - WER: {metrics['wer']:.2f}%, CER: {metrics['cer']:.2f}%, RTF: {speed_metrics['rtf']:.3f}")
        
        return results
    
    def summarize_results(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Summarize benchmark results by model.
        
        Args:
            results_df: Results DataFrame
            
        Returns:
            Summary DataFrame
        """
        summary = []
        
        for model in results_df["model"].unique():
            model_df = results_df[results_df["model"] == model]
            
            # Get metrics (same for all rows of same model)
            first_row = model_df.iloc[0]
            
            summary.append({
                "model": model,
                "samples": len(model_df),
                "wer": first_row.get("wer", np.nan),
                "cer": first_row.get("cer", np.nan),
                "mer": first_row.get("mer", np.nan),
                "wil": first_row.get("wil", np.nan),
                "avg_inference_time_sec": first_row.get("avg_inference_time_sec", np.nan),
                "rtf": first_row.get("rtf", np.nan),
                "samples_per_second": first_row.get("samples_per_second", np.nan),
            })
        
        summary_df = pd.DataFrame(summary)
        
        # Sort by WER
        summary_df = summary_df.sort_values("wer")
        
        return summary_df


def run_baseline_benchmark(
    manifest_path: Union[str, Path],
    output_path: Union[str, Path],
    models: Optional[List[str]] = None,
    max_samples: Optional[int] = None,
    device: Optional[str] = None,
) -> pd.DataFrame:
    """Run baseline benchmark from CLI.
    
    Args:
        manifest_path: Path to test manifest
        output_path: Path to save results
        models: List of models to benchmark (default: whisper-large-v3-turbo, wav2vec2-xlsr-german)
        max_samples: Max samples for quick test
        device: Device to use
        
    Returns:
        Results DataFrame
    """
    if models is None:
        # Default models to compare
        models = [
            "whisper-large-v3-turbo",
            "whisper-medium",
            "wav2vec2-xlsr-german",
        ]
    
    benchmark = ModelBenchmark(models, device=device)
    
    results_df = benchmark.run_benchmark(
        manifest_path=manifest_path,
        output_path=output_path,
        max_samples=max_samples,
    )
    
    # Print summary
    summary = benchmark.summarize_results(results_df)
    logger.info("\nBenchmark Summary:")
    logger.info(f"\n{summary.to_string()}")
    
    return results_df

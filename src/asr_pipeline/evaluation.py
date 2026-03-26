"""Evaluation module for comparing baseline and fine-tuned models."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import jiwer
import numpy as np
import pandas as pd
from tqdm import tqdm

from asr_pipeline.models import load_model
from asr_pipeline.utils import get_logger, load_manifest, save_manifest

logger = get_logger(__name__)


class ModelEvaluator:
    """Evaluate ASR models."""
    
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
    ):
        """Initialize evaluator.
        
        Args:
            model_path: Path to model or model name
            device: Device to use
        """
        self.model_path = model_path
        self.device = device
        self.model = None
    
    def load(self) -> None:
        """Load the model."""
        self.model = load_model(self.model_path, device=self.device)
        self.model.load()
    
    def evaluate(
        self,
        manifest_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        transcript_col: str = "transcript_normalized",
        max_samples: Optional[int] = None,
    ) -> Dict:
        """Evaluate model on test set.
        
        Args:
            manifest_path: Path to test manifest
            output_path: Optional path to save results
            transcript_col: Column with reference transcripts
            max_samples: Maximum samples to evaluate
            
        Returns:
            Evaluation results dict
        """
        if self.model is None:
            self.load()
        
        df = load_manifest(manifest_path)
        
        # Filter to valid samples
        if "is_valid" in df.columns:
            df = df[df["is_valid"]]
        
        if transcript_col not in df.columns:
            transcript_col = "transcript_raw"
        
        df = df[df[transcript_col].notna()]
        
        if max_samples:
            df = df.sample(min(max_samples, len(df)), random_state=42)
        
        logger.info(f"Evaluating on {len(df)} samples")
        
        predictions = []
        references = []
        inference_times = []
        errors = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
            audio_path = row["audio_path"]
            reference = row[transcript_col]
            
            try:
                result = self.model.transcribe(audio_path)
                prediction = result["transcription"]
                inference_time = result["inference_time_sec"]
                
                predictions.append(prediction)
                references.append(reference)
                inference_times.append(inference_time)
                
                # Compute per-sample WER
                sample_wer = jiwer.wer([reference], [prediction]) * 100
                errors.append({
                    "sample_id": row.get("sample_id", idx),
                    "reference": reference,
                    "prediction": prediction,
                    "wer": sample_wer,
                    "inference_time_sec": inference_time,
                })
                
            except Exception as e:
                logger.warning(f"Failed to transcribe {audio_path}: {e}")
        
        # Compute overall metrics
        if predictions:
            overall_wer = jiwer.wer(references, predictions) * 100
            overall_cer = jiwer.cer(references, predictions) * 100
            
            total_duration = df["duration_sec"].sum() if "duration_sec" in df.columns else 0
            total_inference_time = sum(inference_times)
            
            results = {
                "model_path": self.model_path,
                "num_samples": len(predictions),
                "wer": overall_wer,
                "cer": overall_cer,
                "avg_inference_time_sec": np.mean(inference_times),
                "total_inference_time_sec": total_inference_time,
                "rtf": total_inference_time / total_duration if total_duration > 0 else 0,
                "samples_per_second": len(predictions) / total_inference_time if total_inference_time > 0 else 0,
                "error_details": errors,
            }
            
            logger.info(f"Evaluation complete: WER={overall_wer:.2f}%, CER={overall_cer:.2f}%")
        else:
            results = {
                "model_path": self.model_path,
                "error": "No successful predictions",
            }
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save JSON results
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
            
            # Save error details as CSV
            if errors:
                errors_df = pd.DataFrame(errors)
                errors_csv_path = output_path.with_suffix(".errors.csv")
                errors_df.to_csv(errors_csv_path, index=False)
        
        return results


class ComparisonReport:
    """Compare baseline and fine-tuned models."""
    
    def __init__(
        self,
        baseline_results: Dict,
        finetuned_results: Dict,
    ):
        """Initialize comparison.
        
        Args:
            baseline_results: Baseline model results
            finetuned_results: Fine-tuned model results
        """
        self.baseline = baseline_results
        self.finetuned = finetuned_results
    
    def compute_improvement(
        self,
        metric: str,
    ) -> Dict[str, float]:
        """Compute improvement for a metric.
        
        Args:
            metric: Metric name
            
        Returns:
            Dict with absolute and relative improvement
        """
        baseline_val = self.baseline.get(metric, 0)
        finetuned_val = self.finetuned.get(metric, 0)
        
        absolute = baseline_val - finetuned_val  # For error rates, lower is better
        relative = (absolute / baseline_val * 100) if baseline_val > 0 else 0
        
        return {
            "baseline": baseline_val,
            "finetuned": finetuned_val,
            "absolute_improvement": absolute,
            "relative_improvement_pct": relative,
        }
    
    def to_dict(self) -> Dict:
        """Convert comparison to dict.
        
        Returns:
            Comparison dict
        """
        metrics = ["wer", "cer", "avg_inference_time_sec", "rtf"]
        
        comparison = {
            "baseline_model": self.baseline.get("model_path"),
            "finetuned_model": self.finetuned.get("model_path"),
            "metrics": {},
        }
        
        for metric in metrics:
            comparison["metrics"][metric] = self.compute_improvement(metric)
        
        return comparison
    
    def to_markdown(self) -> str:
        """Generate markdown report.
        
        Returns:
            Markdown string
        """
        lines = [
            "# Model Evaluation Report\n",
            "## Comparison: Baseline vs Fine-tuned\n",
        ]
        
        lines.extend([
            f"**Baseline Model**: `{self.baseline.get('model_path')}`\n",
            f"**Fine-tuned Model**: `{self.finetuned.get('model_path')}`\n",
            "\n",
        ])
        
        lines.extend([
            "## Metrics Comparison\n",
            "| Metric | Baseline | Fine-tuned | Absolute Improvement | Relative Improvement |\n",
            "|--------|----------|------------|---------------------|---------------------|\n",
        ])
        
        for metric, values in self.to_dict()["metrics"].items():
            lines.append(
                f"| {metric.upper()} | "
                f"{values['baseline']:.2f} | "
                f"{values['finetuned']:.2f} | "
                f"{values['absolute_improvement']:.2f} | "
                f"{values['relative_improvement_pct']:.1f}% |\n"
            )
        
        lines.append("\n")
        
        # Error analysis
        lines.extend([
            "## Error Analysis\n",
            "### Worst Predictions (Baseline)\n",
        ])
        
        baseline_errors = self.baseline.get("error_details", [])
        if baseline_errors:
            worst_baseline = sorted(baseline_errors, key=lambda x: x["wer"], reverse=True)[:5]
            for err in worst_baseline:
                lines.extend([
                    f"- **WER**: {err['wer']:.1f}%\n",
                    f"  - Reference: `{err['reference']}`\n",
                    f"  - Prediction: `{err['prediction']}`\n",
                ])
        
        lines.append("\n")
        
        return "".join(lines)
    
    def save(self, output_path: Union[str, Path]) -> None:
        """Save report.
        
        Args:
            output_path: Output path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save markdown
        markdown_path = output_path.with_suffix(".md")
        with open(markdown_path, "w") as f:
            f.write(self.to_markdown())
        
        # Save JSON
        json_path = output_path.with_suffix(".json")
        with open(json_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        
        logger.info(f"Saved comparison report to {output_path}")


def run_evaluation(
    model_path: Union[str, Path],
    test_manifest: Union[str, Path],
    output_path: Union[str, Path],
    baseline_results_path: Optional[Union[str, Path]] = None,
    max_samples: Optional[int] = None,
    device: Optional[str] = None,
) -> Dict:
    """Run evaluation from CLI.
    
    Args:
        model_path: Path to model or model name
        test_manifest: Path to test manifest
        output_path: Path to save results
        baseline_results_path: Optional baseline results for comparison
        max_samples: Maximum samples to evaluate
        device: Device to use
        
    Returns:
        Evaluation results
    """
    evaluator = ModelEvaluator(model_path, device=device)
    results = evaluator.evaluate(
        manifest_path=test_manifest,
        output_path=output_path,
        max_samples=max_samples,
    )
    
    # Compare with baseline if provided
    if baseline_results_path:
        with open(baseline_results_path, "r") as f:
            baseline_results = json.load(f)
        
        comparison = ComparisonReport(baseline_results, results)
        comparison_path = Path(output_path).parent / "comparison"
        comparison.save(comparison_path)
    
    return results

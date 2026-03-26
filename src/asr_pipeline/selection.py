"""Model selection module for choosing primary and fallback models."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from asr_pipeline.utils import get_logger, load_manifest, save_manifest

logger = get_logger(__name__)


class ModelSelector:
    """Select best models based on benchmark results."""
    
    def __init__(
        self,
        wer_weight: float = 0.5,
        cer_weight: float = 0.2,
        speed_weight: float = 0.2,
        stability_weight: float = 0.1,
        max_rtf: float = 1.0,
    ):
        """Initialize model selector.
        
        Args:
            wer_weight: Weight for WER in scoring
            cer_weight: Weight for CER in scoring
            speed_weight: Weight for speed (samples/sec) in scoring
            stability_weight: Weight for stability in scoring
            max_rtf: Maximum acceptable RTF (real-time factor)
        """
        self.wer_weight = wer_weight
        self.cer_weight = cer_weight
        self.speed_weight = speed_weight
        self.stability_weight = stability_weight
        self.max_rtf = max_rtf
        
        # Validate weights
        total = wer_weight + cer_weight + speed_weight + stability_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {total}")
    
    def compute_composite_score(
        self,
        row: pd.Series,
    ) -> float:
        """Compute composite score for a model.
        
        Lower is better (like error rates).
        
        Args:
            row: DataFrame row with metrics
            
        Returns:
            Composite score
        """
        # Normalize WER (0-100 scale, lower is better)
        wer_score = row["wer"] / 100.0
        
        # Normalize CER (0-100 scale)
        cer_score = row["cer"] / 100.0
        
        # Normalize speed (higher is better, so invert)
        # Assume max reasonable speed is 100 samples/sec
        speed = row.get("samples_per_second", 0)
        speed_score = 1.0 - min(speed / 100.0, 1.0)
        
        # Stability score (based on error rate variance if available, else 0)
        stability_score = 0.0  # Placeholder
        
        composite = (
            self.wer_weight * wer_score +
            self.cer_weight * cer_score +
            self.speed_weight * speed_score +
            self.stability_weight * stability_score
        )
        
        return composite
    
    def filter_candidates(
        self,
        summary_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Filter models based on hard constraints.
        
        Args:
            summary_df: Summary DataFrame
            
        Returns:
            Filtered DataFrame
        """
        # Filter by RTF constraint
        if "rtf" in summary_df.columns:
            candidates = summary_df[summary_df["rtf"] <= self.max_rtf].copy()
            if len(candidates) == 0:
                logger.warning(f"No models meet RTF constraint of {self.max_rtf}, using all")
                candidates = summary_df.copy()
        else:
            candidates = summary_df.copy()
        
        return candidates
    
    def select_models(
        self,
        benchmark_results_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Dict]:
        """Select primary and fallback models.
        
        Args:
            benchmark_results_path: Path to benchmark results
            output_path: Optional path to save selection
            
        Returns:
            Dict with selection results
        """
        results_df = load_manifest(benchmark_results_path)
        
        # Create summary
        summary = []
        for model in results_df["model"].unique():
            model_df = results_df[results_df["model"] == model]
            first_row = model_df.iloc[0]
            
            summary.append({
                "model": model,
                "samples": len(model_df),
                "wer": first_row.get("wer", np.nan),
                "cer": first_row.get("cer", np.nan),
                "avg_inference_time_sec": first_row.get("avg_inference_time_sec", np.nan),
                "rtf": first_row.get("rtf", np.nan),
                "samples_per_second": first_row.get("samples_per_second", np.nan),
            })
        
        summary_df = pd.DataFrame(summary)
        
        # Filter candidates
        candidates = self.filter_candidates(summary_df)
        
        # Compute composite scores
        candidates["composite_score"] = candidates.apply(
            self.compute_composite_score,
            axis=1,
        )
        
        # Sort by composite score
        candidates = candidates.sort_values("composite_score")
        
        # Select primary: best composite score
        primary = candidates.iloc[0].to_dict() if len(candidates) > 0 else None
        
        # Select fallback: best WER among remaining, but different from primary
        fallback = None
        if len(candidates) > 1:
            remaining = candidates[candidates["model"] != primary["model"]]
            if len(remaining) > 0:
                fallback = remaining.iloc[0].to_dict()
        
        selection = {
            "primary": primary,
            "fallback": fallback,
            "all_models": candidates.to_dict("records"),
            "selection_criteria": {
                "wer_weight": self.wer_weight,
                "cer_weight": self.cer_weight,
                "speed_weight": self.speed_weight,
                "stability_weight": self.stability_weight,
                "max_rtf": self.max_rtf,
            },
        }
        
        logger.info(f"Selected primary model: {primary['model'] if primary else None}")
        logger.info(f"Selected fallback model: {fallback['model'] if fallback else None}")
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(selection, f, indent=2, default=str)
            logger.info(f"Saved selection to {output_path}")
        
        return selection
    
    def get_selection_summary(
        self,
        selection: Dict,
    ) -> pd.DataFrame:
        """Get summary table of selection.
        
        Args:
            selection: Selection dict
            
        Returns:
            Summary DataFrame
        """
        rows = []
        
        if selection["primary"]:
            row = selection["primary"].copy()
            row["role"] = "primary"
            rows.append(row)
        
        if selection["fallback"]:
            row = selection["fallback"].copy()
            row["role"] = "fallback"
            rows.append(row)
        
        return pd.DataFrame(rows)


class SelectionReport:
    """Generate model selection report."""
    
    def __init__(self, selection: Dict):
        """Initialize report.
        
        Args:
            selection: Selection dict from ModelSelector
        """
        self.selection = selection
    
    def to_markdown(self) -> str:
        """Generate markdown report.
        
        Returns:
            Markdown string
        """
        lines = [
            "# Model Selection Report\n",
            "## Selected Models\n",
        ]
        
        primary = self.selection["primary"]
        fallback = self.selection["fallback"]
        
        if primary:
            lines.extend([
                f"### Primary Model: `{primary['model']}`\n",
                f"- **WER**: {primary['wer']:.2f}%\n",
                f"- **CER**: {primary['cer']:.2f}%\n",
                f"- **RTF**: {primary['rtf']:.3f}\n",
                f"- **Speed**: {primary['samples_per_second']:.2f} samples/sec\n",
                "\n",
            ])
        
        if fallback:
            lines.extend([
                f"### Fallback Model: `{fallback['model']}`\n",
                f"- **WER**: {fallback['wer']:.2f}%\n",
                f"- **CER**: {fallback['cer']:.2f}%\n",
                f"- **RTF**: {fallback['rtf']:.3f}\n",
                f"- **Speed**: {fallback['samples_per_second']:.2f} samples/sec\n",
                "\n",
            ])
        
        lines.extend([
            "## Selection Criteria\n",
            f"- WER Weight: {self.selection['selection_criteria']['wer_weight']}\n",
            f"- CER Weight: {self.selection['selection_criteria']['cer_weight']}\n",
            f"- Speed Weight: {self.selection['selection_criteria']['speed_weight']}\n",
            f"- Max RTF: {self.selection['selection_criteria']['max_rtf']}\n",
            "\n",
        ])
        
        lines.extend([
            "## All Models Ranked\n",
            "| Rank | Model | WER | CER | RTF | Composite Score |\n",
            "|------|-------|-----|-----|-----|------------------|\n",
        ])
        
        for i, model in enumerate(self.selection["all_models"], 1):
            lines.append(
                f"| {i} | {model['model']} | "
                f"{model['wer']:.2f}% | {model['cer']:.2f}% | "
                f"{model['rtf']:.3f} | {model.get('composite_score', 'N/A')} |\n"
            )
        
        return "".join(lines)
    
    def save(self, output_path: Union[str, Path]) -> None:
        """Save report to file.
        
        Args:
            output_path: Output path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        markdown = self.to_markdown()
        
        with open(output_path, "w") as f:
            f.write(markdown)
        
        logger.info(f"Saved selection report to {output_path}")


def run_model_selection(
    benchmark_results_path: Union[str, Path],
    output_path: Union[str, Path],
    report_path: Optional[Union[str, Path]] = None,
    wer_weight: float = 0.5,
    speed_weight: float = 0.2,
) -> Dict:
    """Run model selection from CLI.
    
    Args:
        benchmark_results_path: Path to benchmark results
        output_path: Path to save selection JSON
        report_path: Optional path to save markdown report
        wer_weight: Weight for WER
        speed_weight: Weight for speed
        
    Returns:
        Selection dict
    """
    selector = ModelSelector(
        wer_weight=wer_weight,
        speed_weight=speed_weight,
    )
    
    selection = selector.select_models(benchmark_results_path, output_path)
    
    if report_path:
        report = SelectionReport(selection)
        report.save(report_path)
    
    return selection

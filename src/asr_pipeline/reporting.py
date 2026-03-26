"""Reporting module for generating evaluation reports."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from rich.console import Console
from rich.table import Table

from asr_pipeline.utils import get_logger, load_manifest

logger = get_logger(__name__)
console = Console()


class ReportGenerator:
    """Generate evaluation reports."""
    
    def __init__(self, output_dir: Union[str, Path]):
        """Initialize report generator.
        
        Args:
            output_dir: Directory for output reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams["figure.figsize"] = (12, 6)
    
    def generate_dataset_report(
        self,
        manifest_path: Union[str, Path],
    ) -> Path:
        """Generate dataset statistics report.
        
        Args:
            manifest_path: Path to manifest
            
        Returns:
            Path to generated report
        """
        df = load_manifest(manifest_path)
        
        report_path = self.output_dir / "dataset_report.md"
        
        lines = [
            "# Dataset Report\n",
            f"**Source**: {manifest_path}\n",
            f"**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n",
        ]
        
        # Basic stats
        lines.extend([
            "## Overview\n",
            f"- **Total Samples**: {len(df):,}\n",
        ])
        
        if "is_valid" in df.columns:
            valid_count = df["is_valid"].sum()
            lines.append(f"- **Valid Samples**: {valid_count:,} ({valid_count/len(df)*100:.1f}%)\n")
        
        # Duration stats
        if "duration_sec" in df.columns:
            total_hours = df["duration_sec"].sum() / 3600
            lines.extend([
                "\n## Duration Statistics\n",
                f"- **Total Duration**: {total_hours:.2f} hours\n",
                f"- **Mean Duration**: {df['duration_sec'].mean():.2f} seconds\n",
                f"- **Median Duration**: {df['duration_sec'].median():.2f} seconds\n",
                f"- **Min Duration**: {df['duration_sec'].min():.2f} seconds\n",
                f"- **Max Duration**: {df['duration_sec'].max():.2f} seconds\n",
            ])
            
            # Duration distribution plot
            fig, ax = plt.subplots()
            sns.histplot(df["duration_sec"], bins=50, kde=True, ax=ax)
            ax.set_xlabel("Duration (seconds)")
            ax.set_ylabel("Count")
            ax.set_title("Duration Distribution")
            plot_path = self.output_dir / "duration_distribution.png"
            fig.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            
            lines.append(f"\n![Duration Distribution]({plot_path.name})\n")
        
        # Split distribution
        if "split" in df.columns:
            lines.extend(["\n## Split Distribution\n"])
            split_counts = df["split"].value_counts()
            for split, count in split_counts.items():
                lines.append(f"- **{split}**: {count:,} samples\n")
        
        # Domain distribution
        if "domain" in df.columns:
            lines.extend(["\n## Domain Distribution\n"])
            domain_counts = df["domain"].value_counts()
            for domain, count in domain_counts.items():
                lines.append(f"- **{domain}**: {count:,} samples\n")
        
        # Write report
        with open(report_path, "w") as f:
            f.writelines(lines)
        
        logger.info(f"Generated dataset report: {report_path}")
        return report_path
    
    def generate_benchmark_report(
        self,
        benchmark_results_path: Union[str, Path],
    ) -> Path:
        """Generate benchmark comparison report.
        
        Args:
            benchmark_results_path: Path to benchmark results
            
        Returns:
            Path to generated report
        """
        df = load_manifest(benchmark_results_path)
        
        report_path = self.output_dir / "benchmark_report.md"
        
        lines = [
            "# Benchmark Report\n",
            f"**Source**: {benchmark_results_path}\n",
            f"**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n",
        ]
        
        # Summary table
        lines.extend(["## Model Comparison\n"])
        
        summary_data = []
        for model in df["model"].unique():
            model_df = df[df["model"] == model]
            first_row = model_df.iloc[0]
            
            summary_data.append({
                "Model": model,
                "WER (%)": f"{first_row.get('wer', 0):.2f}",
                "CER (%)": f"{first_row.get('cer', 0):.2f}",
                "RTF": f"{first_row.get('rtf', 0):.3f}",
                "Samples/sec": f"{first_row.get('samples_per_second', 0):.1f}",
            })
        
        summary_df = pd.DataFrame(summary_data)
        lines.append(summary_df.to_markdown(index=False))
        lines.append("\n\n")
        
        # WER comparison plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Get first row per model for metrics
        summary_rows = []
        for model in df["model"].unique():
            model_df = df[df["model"] == model]
            summary_rows.append(model_df.iloc[0])
        
        summary_metrics = pd.DataFrame(summary_rows)
        
        # WER comparison
        ax1 = axes[0]
        sns.barplot(data=summary_metrics, x="model", y="wer", ax=ax1)
        ax1.set_xlabel("Model")
        ax1.set_ylabel("WER (%)")
        ax1.set_title("Word Error Rate Comparison")
        ax1.tick_params(axis="x", rotation=45)
        
        # Speed comparison
        ax2 = axes[1]
        sns.barplot(data=summary_metrics, x="model", y="rtf", ax=ax2)
        ax2.set_xlabel("Model")
        ax2.set_ylabel("RTF (lower is better)")
        ax2.set_title("Real-Time Factor Comparison")
        ax2.tick_params(axis="x", rotation=45)
        
        plt.tight_layout()
        plot_path = self.output_dir / "benchmark_comparison.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        
        lines.append(f"![Benchmark Comparison]({plot_path.name})\n")
        
        # Write report
        with open(report_path, "w") as f:
            f.writelines(lines)
        
        logger.info(f"Generated benchmark report: {report_path}")
        return report_path
    
    def generate_evaluation_report(
        self,
        evaluation_results_path: Union[str, Path],
        baseline_results_path: Optional[Union[str, Path]] = None,
    ) -> Path:
        """Generate evaluation report.
        
        Args:
            evaluation_results_path: Path to evaluation results
            baseline_results_path: Optional path to baseline results
            
        Returns:
            Path to generated report
        """
        with open(evaluation_results_path, "r") as f:
            eval_results = json.load(f)
        
        report_path = self.output_dir / "evaluation_report.md"
        
        lines = [
            "# Evaluation Report\n",
            f"**Model**: {eval_results.get('model_path', 'Unknown')}\n",
            f"**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n",
        ]
        
        # Metrics
        lines.extend([
            "## Performance Metrics\n",
            f"- **WER**: {eval_results.get('wer', 0):.2f}%\n",
            f"- **CER**: {eval_results.get('cer', 0):.2f}%\n",
            f"- **Average Inference Time**: {eval_results.get('avg_inference_time_sec', 0):.3f} seconds\n",
            f"- **RTF**: {eval_results.get('rtf', 0):.3f}\n",
            f"- **Samples/sec**: {eval_results.get('samples_per_second', 0):.1f}\n",
        ])
        
        # Comparison with baseline
        if baseline_results_path:
            with open(baseline_results_path, "r") as f:
                baseline_results = json.load(f)
            
            lines.extend(["\n## Comparison with Baseline\n"])
            
            wer_improvement = baseline_results.get("wer", 0) - eval_results.get("wer", 0)
            cer_improvement = baseline_results.get("cer", 0) - eval_results.get("cer", 0)
            
            lines.extend([
                f"- **WER Improvement**: {wer_improvement:.2f}% "
                f"({baseline_results.get('wer', 0):.2f}% → {eval_results.get('wer', 0):.2f}%)\n",
                f"- **CER Improvement**: {cer_improvement:.2f}% "
                f"({baseline_results.get('cer', 0):.2f}% → {eval_results.get('cer', 0):.2f}%)\n",
            ])
        
        # Error analysis
        error_details = eval_results.get("error_details", [])
        if error_details:
            lines.extend(["\n## Error Analysis\n"])
            lines.append("### Worst Predictions\n")
            
            worst_errors = sorted(error_details, key=lambda x: x["wer"], reverse=True)[:10]
            
            for i, err in enumerate(worst_errors, 1):
                lines.extend([
                    f"{i}. **WER**: {err['wer']:.1f}%\n",
                    f"   - **Reference**: {err['reference']}\n",
                    f"   - **Prediction**: {err['prediction']}\n",
                ])
        
        # Write report
        with open(report_path, "w") as f:
            f.writelines(lines)
        
        logger.info(f"Generated evaluation report: {report_path}")
        return report_path
    
    def generate_inference_report(
        self,
        inference_results_path: Union[str, Path],
    ) -> Path:
        """Generate inference report.
        
        Args:
            inference_results_path: Path to inference results
            
        Returns:
            Path to generated report
        """
        df = load_manifest(inference_results_path)
        
        report_path = self.output_dir / "inference_report.md"
        
        lines = [
            "# Inference Report\n",
            f"**Source**: {inference_results_path}\n",
            f"**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n\n",
        ]
        
        # Overview
        lines.extend([
            "## Overview\n",
            f"- **Total Samples**: {len(df):,}\n",
        ])
        
        if "model_used" in df.columns:
            primary_count = (df["model_used"] == "primary").sum()
            fallback_count = (df["model_used"] == "fallback").sum()
            error_count = (df["model_used"] == "none").sum()
            
            lines.extend([
                f"- **Primary Model**: {primary_count:,} samples\n",
                f"- **Fallback Model**: {fallback_count:,} samples\n",
                f"- **Failed**: {error_count:,} samples\n",
            ])
        
        # Inference time stats
        if "inference_time_sec" in df.columns:
            lines.extend([
                "\n## Inference Speed\n",
                f"- **Mean Time**: {df['inference_time_sec'].mean():.3f} seconds\n",
                f"- **Median Time**: {df['inference_time_sec'].median():.3f} seconds\n",
                f"- **Total Time**: {df['inference_time_sec'].sum():.1f} seconds\n",
            ])
        
        # Dataset WER if available
        if "dataset_wer" in df.columns:
            wer = df["dataset_wer"].iloc[0]
            lines.extend([f"\n## Dataset Quality\n", f"- **WER**: {wer:.2f}%\n"])
        
        # Sample predictions
        lines.extend(["\n## Sample Predictions\n"])
        
        sample_cols = ["sample_id", "predicted_transcript"]
        if "transcript_normalized" in df.columns:
            sample_cols.insert(2, "transcript_normalized")
        elif "transcript_raw" in df.columns:
            sample_cols.insert(2, "transcript_raw")
        
        available_cols = [c for c in sample_cols if c in df.columns]
        samples_df = df[available_cols].head(10)
        
        lines.append(samples_df.to_markdown(index=False))
        
        # Write report
        with open(report_path, "w") as f:
            f.writelines(lines)
        
        logger.info(f"Generated inference report: {report_path}")
        return report_path


def print_summary_table(data: Dict[str, Dict], title: str = "Summary") -> None:
    """Print a summary table to console.
    
    Args:
        data: Dict of row name to dict of column values
        title: Table title
    """
    console.print(f"\n[bold cyan]{title}[/bold cyan]")
    
    table = Table(show_header=True, header_style="bold magenta")
    
    # Add columns
    table.add_column("Item")
    first_row = next(iter(data.values()))
    for col in first_row.keys():
        table.add_column(col)
    
    # Add rows
    for name, values in data.items():
        row_values = [str(v) for v in values.values()]
        table.add_row(name, *row_values)
    
    console.print(table)
    console.print()


def generate_all_reports(
    artifacts_dir: Union[str, Path],
    output_dir: Union[str, Path],
) -> List[Path]:
    """Generate all available reports.
    
    Args:
        artifacts_dir: Directory with artifact files
        output_dir: Directory for reports
        
    Returns:
        List of generated report paths
    """
    artifacts_dir = Path(artifacts_dir)
    output_dir = Path(output_dir)
    
    generator = ReportGenerator(output_dir)
    generated = []
    
    # Dataset report
    manifest_path = artifacts_dir / "manifest_with_splits.parquet"
    if manifest_path.exists():
        generated.append(generator.generate_dataset_report(manifest_path))
    
    # Benchmark report
    benchmark_path = artifacts_dir / "benchmark_results.parquet"
    if benchmark_path.exists():
        generated.append(generator.generate_benchmark_report(benchmark_path))
    
    # Evaluation report
    eval_path = artifacts_dir / "evaluation_results.json"
    if eval_path.exists():
        baseline_path = artifacts_dir / "baseline_results.json"
        baseline_path = baseline_path if baseline_path.exists() else None
        generated.append(generator.generate_evaluation_report(eval_path, baseline_path))
    
    # Inference report
    inference_path = artifacts_dir / "inference_results.parquet"
    if inference_path.exists():
        generated.append(generator.generate_inference_report(inference_path))
    
    return generated

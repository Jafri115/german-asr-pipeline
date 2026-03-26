"""Command-line interface for the ASR pipeline."""

from pathlib import Path
from typing import Optional

import click
from omegaconf import OmegaConf

from asr_pipeline.benchmark import run_baseline_benchmark
from asr_pipeline.evaluation import run_evaluation
from asr_pipeline.finetune import run_finetune
from asr_pipeline.ingestion import run_ingestion
from asr_pipeline.inference import run_full_inference
from asr_pipeline.preprocessing import run_preprocessing
from asr_pipeline.selection import run_model_selection
from asr_pipeline.split import run_split
from asr_pipeline.validation import run_validation


@click.group()
def cli():
    """German ASR Pipeline CLI."""
    pass


@cli.command()
@click.option("--audio-dir", required=True, type=click.Path(exists=True), help="Directory with audio files")
@click.option("--output", required=True, type=click.Path(), help="Output manifest path")
@click.option("--transcript-dir", type=click.Path(exists=True), help="Directory with transcripts")
@click.option("--recursive/--no-recursive", default=True, help="Scan recursively")
def ingest(
    audio_dir: str,
    output: str,
    transcript_dir: Optional[str],
    recursive: bool,
):
    """Run data ingestion."""
    run_ingestion(
        audio_dir=audio_dir,
        output_path=output,
        transcript_dir=transcript_dir,
        recursive=recursive,
    )


@cli.command()
@click.option("--manifest", required=True, type=click.Path(exists=True), help="Input manifest")
@click.option("--output", required=True, type=click.Path(), help="Output manifest path")
@click.option("--min-duration", default=1.0, help="Minimum duration in seconds")
@click.option("--max-duration", default=60.0, help="Maximum duration in seconds")
@click.option("--target-sr", default=16000, help="Target sample rate")
@click.option("--require-transcript/--no-require-transcript", default=True, help="Require transcripts")
def validate(
    manifest: str,
    output: str,
    min_duration: float,
    max_duration: float,
    target_sr: int,
    require_transcript: bool,
):
    """Run data validation."""
    run_validation(
        manifest_path=manifest,
        output_path=output,
        min_duration_sec=min_duration,
        max_duration_sec=max_duration,
        target_sample_rate=target_sr,
        require_transcript=require_transcript,
    )


@cli.command()
@click.option("--manifest", required=True, type=click.Path(exists=True), help="Input manifest")
@click.option("--output-dir", required=True, type=click.Path(), help="Output directory")
@click.option("--output-manifest", required=True, type=click.Path(), help="Output manifest path")
@click.option("--target-sr", default=16000, help="Target sample rate")
@click.option("--normalize/--no-normalize", default=True, help="Normalize transcripts")
@click.option("--trim-silence/--no-trim-silence", default=False, help="Trim silence")
def preprocess(
    manifest: str,
    output_dir: str,
    output_manifest: str,
    target_sr: int,
    normalize: bool,
    trim_silence: bool,
):
    """Run preprocessing."""
    run_preprocessing(
        manifest_path=manifest,
        output_dir=output_dir,
        output_manifest_path=output_manifest,
        target_sample_rate=target_sr,
        normalize_transcripts=normalize,
        trim_silence=trim_silence,
    )


@cli.command()
@click.option("--manifest", required=True, type=click.Path(exists=True), help="Input manifest")
@click.option("--output-dir", required=True, type=click.Path(), help="Output directory")
@click.option("--train-ratio", default=0.8, help="Training ratio")
@click.option("--val-ratio", default=0.1, help="Validation ratio")
@click.option("--test-ratio", default=0.1, help="Test ratio")
@click.option("--strategy", default="random", type=click.Choice(["random", "stratified", "group", "duration"]), help="Split strategy")
@click.option("--stratify-by", help="Column to stratify by")
@click.option("--group-by", help="Column to group by")
@click.option("--seed", default=42, help="Random seed")
def split(
    manifest: str,
    output_dir: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    strategy: str,
    stratify_by: Optional[str],
    group_by: Optional[str],
    seed: int,
):
    """Create train/val/test splits."""
    run_split(
        manifest_path=manifest,
        output_dir=output_dir,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        split_strategy=strategy,
        stratify_by=stratify_by,
        group_by=group_by,
        random_seed=seed,
    )


@cli.command()
@click.option("--manifest", required=True, type=click.Path(exists=True), help="Test manifest")
@click.option("--output", required=True, type=click.Path(), help="Output path")
@click.option("--models", multiple=True, default=["whisper-large-v3-turbo", "wav2vec2-xlsr-german"], help="Models to benchmark")
@click.option("--max-samples", type=int, help="Maximum samples for quick test")
@click.option("--device", help="Device (cuda/cpu)")
def baseline(
    manifest: str,
    output: str,
    models: tuple,
    max_samples: Optional[int],
    device: Optional[str],
):
    """Run baseline benchmark."""
    run_baseline_benchmark(
        manifest_path=manifest,
        output_path=output,
        models=list(models),
        max_samples=max_samples,
        device=device,
    )


@cli.command()
@click.option("--benchmark-results", required=True, type=click.Path(exists=True), help="Benchmark results")
@click.option("--output", required=True, type=click.Path(), help="Output JSON path")
@click.option("--report", type=click.Path(), help="Output markdown report path")
@click.option("--wer-weight", default=0.5, help="WER weight in scoring")
@click.option("--speed-weight", default=0.2, help="Speed weight in scoring")
def select(
    benchmark_results: str,
    output: str,
    report: Optional[str],
    wer_weight: float,
    speed_weight: float,
):
    """Run model selection."""
    run_model_selection(
        benchmark_results_path=benchmark_results,
        output_path=output,
        report_path=report,
        wer_weight=wer_weight,
        speed_weight=speed_weight,
    )


@cli.command()
@click.option("--model", required=True, help="Model name or path")
@click.option("--train-manifest", required=True, type=click.Path(exists=True), help="Training manifest")
@click.option("--val-manifest", required=True, type=click.Path(exists=True), help="Validation manifest")
@click.option("--output-dir", required=True, type=click.Path(), help="Output directory")
@click.option("--epochs", default=3, help="Number of epochs")
@click.option("--batch-size", default=8, help="Batch size")
@click.option("--lr", default=1e-5, help="Learning rate")
def finetune(
    model: str,
    train_manifest: str,
    val_manifest: str,
    output_dir: str,
    epochs: int,
    batch_size: int,
    lr: float,
):
    """Run fine-tuning."""
    run_finetune(
        model_name=model,
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        output_dir=output_dir,
        num_epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
    )


@cli.command()
@click.option("--model", required=True, help="Model path or name")
@click.option("--test-manifest", required=True, type=click.Path(exists=True), help="Test manifest")
@click.option("--output", required=True, type=click.Path(), help="Output path")
@click.option("--baseline-results", type=click.Path(exists=True), help="Baseline results for comparison")
@click.option("--max-samples", type=int, help="Maximum samples to evaluate")
@click.option("--device", help="Device (cuda/cpu)")
def evaluate(
    model: str,
    test_manifest: str,
    output: str,
    baseline_results: Optional[str],
    max_samples: Optional[int],
    device: Optional[str],
):
    """Run evaluation."""
    run_evaluation(
        model_path=model,
        test_manifest=test_manifest,
        output_path=output,
        baseline_results_path=baseline_results,
        max_samples=max_samples,
        device=device,
    )


@cli.command()
@click.option("--model", required=True, help="Primary model path or name")
@click.option("--manifest", required=True, type=click.Path(exists=True), help="Input manifest")
@click.option("--output", required=True, type=click.Path(), help="Output path")
@click.option("--fallback-model", help="Fallback model path")
@click.option("--device", help="Device (cuda/cpu)")
def infer(
    model: str,
    manifest: str,
    output: str,
    fallback_model: Optional[str],
    device: Optional[str],
):
    """Run full inference."""
    run_full_inference(
        model_path=model,
        manifest_path=manifest,
        output_path=output,
        fallback_model_path=fallback_model,
        device=device,
    )


if __name__ == "__main__":
    cli()

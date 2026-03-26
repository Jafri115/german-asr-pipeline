# German ASR Pipeline

An end-to-end, modular machine learning pipeline for German Automatic Speech Recognition (ASR) using open-source models.

## Overview

This pipeline provides a complete workflow for:

1. **Data Ingestion** - Scan and index audio files with transcripts
2. **Validation** - Check file integrity, duration, and transcript availability
3. **Preprocessing** - Resample audio, normalize transcripts, chunk long files
4. **Split Creation** - Create train/validation/test splits
5. **Baseline Benchmarking** - Compare multiple open-source ASR models
6. **Model Selection** - Select primary and fallback models based on metrics
7. **Fine-tuning** - Fine-tune the selected model on your data
8. **Evaluation** - Compare baseline vs fine-tuned performance
9. **Full Inference** - Run transcription on the entire dataset

## Recommended Models

### Primary Model: Whisper Large V3 Turbo
- **Model**: `openai/whisper-large-v3-turbo`
- **Why**: Best balance of accuracy and speed for German
  - ~10-12% WER on German benchmarks
  - 216x real-time factor (very fast)
  - Excellent multilingual support
  - Handles code-switching well
  - Robust to various audio conditions

### Fallback Model: Wav2Vec2 XLS-R German
- **Model**: `facebook/wav2vec2-large-xlsr-53-german`
- **Why**: More stable, less prone to hallucinations
  - Good German-specific performance
  - Deterministic CTC-based output
  - Smaller memory footprint
  - Better for fine-tuning on domain-specific data

## Project Structure

```
german_asr_pipeline/
├── README.md                 # This file
├── pyproject.toml           # Python package configuration
├── configs/                 # YAML configuration files
│   ├── pipeline.yaml        # Main pipeline config
│   └── quick_test.yaml      # Quick test config
├── data/                    # Data directory (gitignored)
│   ├── raw/                 # Raw audio and transcripts
│   └── preprocessed/        # Preprocessed audio
├── manifests/               # Manifest files (parquet/csv)
│   ├── splits/              # Train/val/test splits
│   └── *.parquet            # Intermediate manifests
├── artifacts/               # Model artifacts and results
│   ├── finetuned_model/     # Fine-tuned model
│   ├── benchmark_results.parquet
│   ├── model_selection.json
│   ├── evaluation_results.json
│   ├── inference_results.parquet
│   └── reports/             # Generated reports
├── notebooks/               # Jupyter notebooks for exploration
├── scripts/                 # Runnable pipeline stage scripts
│   ├── run_ingestion.py
│   ├── run_validation.py
│   ├── run_preprocessing.py
│   ├── run_split.py
│   ├── run_baselines.py
│   ├── run_model_selection.py
│   ├── run_finetune.py
│   ├── run_evaluation.py
│   └── run_full_inference.py
├── src/asr_pipeline/        # Source code
│   ├── __init__.py
│   ├── cli.py              # Command-line interface
│   ├── utils.py            # Utility functions
│   ├── ingestion.py        # Data ingestion
│   ├── validation.py       # Data validation
│   ├── preprocessing.py    # Audio/text preprocessing
│   ├── split.py            # Train/val/test split
│   ├── models.py           # Model wrappers
│   ├── benchmark.py        # Model benchmarking
│   ├── selection.py        # Model selection
│   ├── finetune.py         # Fine-tuning
│   ├── evaluation.py       # Evaluation
│   ├── inference.py        # Batch inference
│   └── reporting.py        # Report generation
└── tests/                   # Pytest tests
    ├── test_utils.py
    ├── test_preprocessing.py
    └── test_models.py
```

## Installation

### Requirements

- Python 3.9+
- CUDA-capable GPU (recommended for training/inference)
- 16GB+ RAM
- ~50GB disk space (for models and data)

### Setup

```bash
# Clone the repository
cd german_asr_pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package
pip install -e ".[dev]"

# Or install from requirements
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Your Data

Organize your audio files and transcripts:

```
data/raw/
├── audio/
│   ├── speaker_001/
│   │   ├── utterance_001.wav
│   │   └── utterance_002.wav
│   └── speaker_002/
│       └── utterance_003.wav
└── transcripts/
    ├── speaker_001/
    │   ├── utterance_001.txt
    │   └── utterance_002.txt
    └── speaker_002/
        └── utterance_003.txt
```

Transcript files should contain the text transcription (UTF-8 encoded).

### 2. Run the Pipeline

#### Option A: Using CLI Commands

```bash
# 1. Ingest data
asr-ingest --audio-dir data/raw/audio --output manifests/01_raw_manifest.parquet --transcript-dir data/raw/transcripts

# 2. Validate
asr-validate --manifest manifests/01_raw_manifest.parquet --output manifests/02_validated_manifest.parquet

# 3. Preprocess
asr-preprocess --manifest manifests/02_validated_manifest.parquet --output-dir data/preprocessed --output-manifest manifests/03_preprocessed_manifest.parquet

# 4. Create splits
asr-split --manifest manifests/03_preprocessed_manifest.parquet --output-dir manifests/splits

# 5. Run baseline benchmark (quick test on 50 samples)
asr-baseline --manifest manifests/splits/test.parquet --output artifacts/benchmark_results.parquet --max-samples 50

# 6. Select models
asr-select --benchmark-results artifacts/benchmark_results.parquet --output artifacts/model_selection.json --report artifacts/reports/model_selection.md

# 7. Fine-tune (optional, requires GPU)
asr-finetune --model openai/whisper-large-v3-turbo --train-manifest manifests/splits/train.parquet --val-manifest manifests/splits/val.parquet --output-dir artifacts/finetuned_model

# 8. Evaluate
asr-evaluate --model artifacts/finetuned_model/final_model --test-manifest manifests/splits/test.parquet --output artifacts/evaluation_results.json

# 9. Run full inference
asr-infer --model artifacts/finetuned_model/final_model --manifest manifests/03_preprocessed_manifest.parquet --output artifacts/inference_results.parquet
```

#### Option B: Using Scripts with Config

```bash
# Run all stages using config
python scripts/run_ingestion.py --config configs/pipeline.yaml
python scripts/run_validation.py --config configs/pipeline.yaml
python scripts/run_preprocessing.py --config configs/pipeline.yaml
python scripts/run_split.py --config configs/pipeline.yaml
python scripts/run_baselines.py --config configs/pipeline.yaml --max-samples 50
python scripts/run_model_selection.py --config configs/pipeline.yaml
python scripts/run_finetune.py --config configs/pipeline.yaml
python scripts/run_evaluation.py --config configs/pipeline.yaml
python scripts/run_full_inference.py --config configs/pipeline.yaml
```

#### Option C: Quick Test Mode

For rapid iteration on a small subset:

```bash
# Use quick_test.yaml config with limited samples
python scripts/run_baselines.py --config configs/quick_test.yaml
```

## Configuration

Edit `configs/pipeline.yaml` to customize:

```yaml
# Data paths
data:
  raw_audio_dir: "data/raw/audio"
  transcript_dir: "data/raw/transcripts"

# Pipeline settings
pipeline:
  validation:
    min_duration_sec: 1.0
    max_duration_sec: 60.0
  
  split:
    train_ratio: 0.8
    val_ratio: 0.1
    test_ratio: 0.1
    strategy: "random"  # or "stratified", "group", "duration"
  
  model_selection:
    wer_weight: 0.5
    speed_weight: 0.2

# Benchmark models
benchmark:
  models:
    - "whisper-large-v3-turbo"
    - "whisper-medium"
    - "wav2vec2-xlsr-german"

# Fine-tuning
finetune:
  model_name: "openai/whisper-large-v3-turbo"
  num_epochs: 3
  batch_size: 8
  learning_rate: 1.0e-5
```

## Manifest Format

Manifests are stored as Parquet files with the following columns:

| Column | Description |
|--------|-------------|
| `sample_id` | Unique identifier |
| `audio_path` | Path to audio file |
| `audio_path_original` | Original path (if preprocessed) |
| `transcript_raw` | Raw transcript text |
| `transcript_normalized` | Normalized transcript |
| `transcript_path` | Path to transcript file |
| `duration_sec` | Audio duration in seconds |
| `sample_rate` | Sample rate (Hz) |
| `channels` | Number of channels |
| `speaker_id` | Speaker identifier (extracted from path) |
| `domain` | Domain (e.g., medical, news) |
| `language` | Language code (de) |
| `split` | train/val/test |
| `is_valid` | Validation flag |
| `validation_issues` | List of validation issues |
| `chunk_id` | Chunk identifier (if chunked) |
| `parent_audio_id` | Original sample ID (if chunked) |

## Metrics

The pipeline computes the following metrics:

| Metric | Description |
|--------|-------------|
| **WER** | Word Error Rate (lower is better) |
| **CER** | Character Error Rate (lower is better) |
| **MER** | Match Error Rate |
| **WIL** | Word Information Lost |
| **RTF** | Real-Time Factor (inference_time / audio_duration) |
| **Samples/sec** | Throughput |

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=asr_pipeline --cov-report=html

# Run specific test file
pytest tests/test_preprocessing.py

# Run with verbose output
pytest -v
```

## Development

### Code Style

```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Type checking
mypy src/

# Linting
flake8 src/ tests/
```

### Pre-commit Hooks

```bash
pre-commit install
pre-commit run --all-files
```

## Troubleshooting

### Out of Memory

- Reduce `batch_size` in fine-tuning config
- Use a smaller model (e.g., `whisper-medium` instead of `whisper-large-v3`)
- Enable gradient checkpointing (already enabled by default)
- Use mixed precision training (`fp16: true`)

### Slow Inference

- Use `whisper-large-v3-turbo` for faster inference
- Reduce `max_samples` for quick benchmarks
- Use GPU if available (`device: cuda`)

### Poor Transcription Quality

- Check audio quality (sample rate, noise level)
- Verify transcripts are accurate
- Try fine-tuning on domain-specific data
- Consider chunking long audio files

## License

MIT License - See LICENSE file for details.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## Acknowledgments

- [OpenAI Whisper](https://github.com/openai/whisper)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Facebook Wav2Vec2](https://github.com/pytorch/fairseq/tree/main/examples/wav2vec)
- [jiwer](https://github.com/jitsi/jiwer) for WER/CER computation

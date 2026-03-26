# German ASR Pipeline - Architecture Summary

## Project Overview

A modular, end-to-end machine learning pipeline for German Automatic Speech Recognition (ASR) using open-source models.

## Architecture Principles

1. **Modularity**: Each stage is self-contained and runnable independently
2. **Manifest-driven**: All data flows through parquet/CSV manifests
3. **Configurable**: YAML configs control all pipeline parameters
4. **Observable**: Rich logging, metrics, and reports at each stage
5. **Testable**: pytest coverage for core functionality

## Pipeline Stages

```
┌─────────────────┐
│   Raw Audio     │
│  + Transcripts  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│    Ingestion    │────▶│  Raw Manifest   │
│  (scan/index)   │     │                 │
└─────────────────┘     └────────┬────────┘
                                 │
                                 ▼
┌─────────────────┐     ┌─────────────────┐
│   Validation    │────▶│ Validated Manifest
│ (check quality) │     │                 │
└─────────────────┘     └────────┬────────┘
                                 │
                                 ▼
┌─────────────────┐     ┌─────────────────┐
│  Preprocessing  │────▶│Preprocessed Manifest
│ (resample/norm) │     │                 │
└─────────────────┘     └────────┬────────┘
                                 │
                                 ▼
┌─────────────────┐     ┌─────────────────┐
│  Split Creation │────▶│ Train/Val/Test  │
│                 │     │    Manifests    │
└─────────────────┘     └────────┬────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
                    ▼                         ▼
         ┌─────────────────┐      ┌─────────────────┐
         │Baseline Benchmark│      │   Train Split   │
         │                 │      │                 │
         └────────┬────────┘      └────────┬────────┘
                  │                        │
                  ▼                        ▼
         ┌─────────────────┐      ┌─────────────────┐
         │  Model Selection│      │   Fine-tuning   │
         │                 │      │                 │
         └────────┬────────┘      └────────┬────────┘
                  │                        │
                  │              ┌─────────┴─────────┐
                  │              │                   │
                  │              ▼                   ▼
                  │    ┌─────────────────┐  ┌─────────────────┐
                  │    │ Finetuned Model │  │  Evaluation vs  │
                  │    │                 │  │    Baseline     │
                  │    └────────┬────────┘  └────────┬────────┘
                  │             │                    │
                  │             └────────┬───────────┘
                  │                      │
                  ▼                      ▼
         ┌─────────────────┐    ┌─────────────────┐
         │ Primary Model   │    │ Fallback Model  │
         │ (for inference) │    │ (for fallback)  │
         └────────┬────────┘    └─────────────────┘
                  │
                  ▼
         ┌─────────────────┐
         │  Full Inference │
         │                 │
         └────────┬────────┘
                  │
                  ▼
         ┌─────────────────┐
         │  Final Outputs  │
         │  + Reports      │
         └─────────────────┘
```

## Module Descriptions

### Core Modules

| Module | Purpose | Key Classes/Functions |
|--------|---------|----------------------|
| `utils.py` | Common utilities | `get_logger()`, `load_manifest()`, `save_manifest()` |
| `ingestion.py` | Data discovery | `DataIngester.scan_audio_files()`, `create_manifest()` |
| `validation.py` | Quality checks | `DataValidator.validate_manifest()` |
| `preprocessing.py` | Audio/text prep | `AudioPreprocessor`, `TranscriptNormalizer`, `AudioChunker` |
| `split.py` | Dataset splits | `SplitCreator.create_split()` with multiple strategies |

### Model Modules

| Module | Purpose | Key Classes/Functions |
|--------|---------|----------------------|
| `models.py` | Model wrappers | `WhisperModel`, `Wav2Vec2Model`, `ModelRegistry` |
| `benchmark.py` | Model comparison | `ModelBenchmark.run_benchmark()`, `MetricsComputer` |
| `selection.py` | Model selection | `ModelSelector.select_models()` |

### Training & Evaluation Modules

| Module | Purpose | Key Classes/Functions |
|--------|---------|----------------------|
| `finetune.py` | Fine-tuning | `FinetuneTrainer`, `AudioDataset`, `DataCollatorSpeechSeq2Seq` |
| `evaluation.py` | Evaluation | `ModelEvaluator.evaluate()`, `ComparisonReport` |
| `inference.py` | Batch inference | `BatchInference.run_inference()`, `StreamingInference` |
| `reporting.py` | Report generation | `ReportGenerator`, visualization functions |

## Manifest Schema

### Raw Manifest (after ingestion)

```
sample_id: str          # Unique identifier
audio_path: str         # Path to audio file
audio_path_absolute: str # Absolute path
transcript_path: str    # Path to transcript file
transcript_raw: str     # Raw transcript text
duration_sec: float     # Audio duration
sample_rate: int        # Sample rate
channels: int           # Number of channels
speaker_id: str         # Speaker identifier
domain: str            # Domain (medical, news, etc.)
language: str          # Language code (de)
is_valid: bool         # Validation flag
```

### Preprocessed Manifest

Adds to raw manifest:

```
audio_path_original: str    # Original path before preprocessing
transcript_normalized: str  # Normalized transcript
is_preprocessed: bool       # Preprocessing flag
```

### Split Manifest

Adds to preprocessed manifest:

```
split: str              # train/val/test
```

### Inference Results

Adds to split manifest:

```
predicted_transcript: str   # Model prediction
inference_time_sec: float   # Inference time
model_used: str            # primary/fallback/none
inference_error: str       # Error message if failed
dataset_wer: float         # Overall WER
```

## Configuration Structure

```yaml
# Data paths
data:
  raw_audio_dir: str
  transcript_dir: str
  preprocessed_audio_dir: str

# Pipeline paths
paths:
  manifests: {raw, validated, preprocessed, with_splits}
  splits_dir: str
  artifacts: {benchmark_results, model_selection, ...}
  reports_dir: str

# Stage configs
pipeline:
  ingestion: {...}
  validation: {...}
  preprocessing: {...}
  split: {...}
  model_selection: {...}

# Model configs
benchmark: {models, device, max_samples}
finetune: {model_name, num_epochs, batch_size, ...}
evaluation: {device, max_samples}
inference: {device, batch_size, use_fallback}
```

## Key Design Decisions

### 1. Manifest-Driven Architecture

- All data passes through pandas DataFrames saved as Parquet
- Enables easy inspection, debugging, and modification
- Supports both CSV and Parquet formats

### 2. Stage Independence

- Each stage can be run independently
- Stages communicate via manifests
- Easy to re-run specific stages after changes

### 3. Model Abstraction

- `ASRModel` base class for all models
- Consistent interface: `load()`, `transcribe()`, `unload()`
- Easy to add new models

### 4. Configuration Management

- OmegaConf/YAML for hierarchical configs
- Configs can be overridden via CLI
- Separate configs for quick testing

### 5. Error Handling

- Graceful degradation (fallback models)
- Validation issues tracked per sample
- Detailed error logging

## Extensibility

### Adding a New Model

1. Create wrapper class inheriting from `ASRModel`
2. Add to `ModelRegistry.GERMAN_MODELS`
3. Update config with new model key

### Adding a New Pipeline Stage

1. Create module with core functionality
2. Add CLI command in `cli.py`
3. Create runner script in `scripts/`
4. Add config section in YAML

### Adding New Metrics

1. Add metric computation in `MetricsComputer`
2. Update benchmark and evaluation to include metric
3. Update reports to visualize metric

## Performance Considerations

### Memory Management

- Models are unloaded after use (`model.unload()`)
- Gradient checkpointing enabled for training
- Batch processing with configurable batch sizes

### Speed Optimizations

- Mixed precision (fp16) training/inference
- Multi-worker data loading
- Chunking for long audio files

### Disk Usage

- Parquet format for efficient storage
- Intermediate manifests can be deleted after pipeline completion
- Audio files are not duplicated (symlinks or path references)

## Testing Strategy

| Test Type | Coverage | Files |
|-----------|----------|-------|
| Unit tests | Utils, preprocessing | `test_utils.py`, `test_preprocessing.py` |
| Model tests | Model wrappers (mocked) | `test_models.py` |
| Integration | Full pipeline (optional) | Manual testing |

## CLI Commands

```bash
# Data preparation
asr-ingest --audio-dir DIR --output MANIFEST
asr-validate --manifest MANIFEST --output MANIFEST
asr-preprocess --manifest MANIFEST --output-dir DIR --output-manifest MANIFEST
asr-split --manifest MANIFEST --output-dir DIR

# Model selection
asr-baseline --manifest MANIFEST --output RESULTS
asr-select --benchmark-results RESULTS --output SELECTION

# Training and evaluation
asr-finetune --model MODEL --train-manifest MANIFEST --val-manifest MANIFEST --output-dir DIR
asr-evaluate --model MODEL --test-manifest MANIFEST --output RESULTS

# Inference
asr-infer --model MODEL --manifest MANIFEST --output RESULTS
```

## Output Artifacts

| Artifact | Format | Description |
|----------|--------|-------------|
| Manifests | Parquet/CSV | Data at each stage |
| Benchmark Results | Parquet | Per-sample predictions and metrics |
| Model Selection | JSON | Primary/fallback model selection |
| Fine-tuned Model | HF format | Saved model and processor |
| Evaluation Results | JSON | Metrics and error analysis |
| Inference Results | Parquet | Full dataset predictions |
| Reports | Markdown | Human-readable summaries |
| Plots | PNG | Visualizations |

## Dependencies

### Core
- PyTorch + torchaudio
- Hugging Face Transformers
- Datasets

### Audio
- librosa
- soundfile

### Data
- pandas
- pyarrow

### Metrics
- jiwer (WER/CER)
- evaluate

### Config
- omegaconf
- hydra-core

### Visualization
- matplotlib
- seaborn

### CLI
- click
- rich

## Future Enhancements

1. **Streaming inference** for real-time applications
2. **VAD integration** for automatic speech segmentation
3. **Language identification** for multilingual scenarios
4. **Speaker diarization** for multi-speaker audio
5. **Active learning** pipeline for data selection
6. **Model quantization** for edge deployment
7. **ONNX export** for optimized inference
8. **Web UI** for interactive exploration

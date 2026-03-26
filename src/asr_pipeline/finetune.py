"""Fine-tuning module for ASR models."""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    TrainingArguments,
)

from asr_pipeline.utils import get_logger, load_manifest

logger = get_logger(__name__)


@dataclass
class FinetuneConfig:
    """Configuration for fine-tuning."""
    
    model_name: str
    output_dir: str
    train_manifest: str
    val_manifest: str
    
    # Training hyperparameters
    num_epochs: int = 3
    batch_size: int = 8
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1e-5
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    
    # Data processing
    max_input_length: float = 30.0  # seconds
    max_label_length: int = 448
    
    # Logging and saving
    logging_steps: int = 25
    eval_steps: int = 100
    save_steps: int = 100
    save_total_limit: int = 3
    
    # Early stopping
    early_stopping_patience: int = 3
    
    # Device
    fp16: bool = True
    dataloader_num_workers: int = 4
    
    # Seed
    seed: int = 42


class AudioDataset:
    """Prepare dataset for fine-tuning."""
    
    def __init__(
        self,
        processor,
        max_input_length: float = 30.0,
        max_label_length: int = 448,
    ):
        """Initialize dataset.
        
        Args:
            processor: HuggingFace processor
            max_input_length: Max audio length in seconds
            max_label_length: Max label length
        """
        self.processor = processor
        self.max_input_length = max_input_length
        self.max_label_length = max_label_length
        self.sampling_rate = processor.feature_extractor.sampling_rate
    
    def load_manifest_to_dataset(self, manifest_path: Union[str, Path]) -> Dataset:
        """Load manifest to HuggingFace Dataset.
        
        Args:
            manifest_path: Path to manifest
            
        Returns:
            HuggingFace Dataset
        """
        df = load_manifest(manifest_path)
        
        # Filter to valid samples
        if "is_valid" in df.columns:
            df = df[df["is_valid"]]
        
        # Select relevant columns
        columns = ["audio_path", "transcript_normalized", "transcript_raw"]
        available_cols = [c for c in columns if c in df.columns]
        df = df[available_cols]
        
        # Rename columns
        if "transcript_normalized" in df.columns:
            df = df.rename(columns={"transcript_normalized": "sentence"})
        else:
            df = df.rename(columns={"transcript_raw": "sentence"})
        
        df = df.rename(columns={"audio_path": "audio"})
        
        # Convert to Dataset
        dataset = Dataset.from_pandas(df)
        
        return dataset
    
    def prepare_dataset(self, dataset: Dataset) -> Dataset:
        """Prepare dataset with audio processing.
        
        Args:
            dataset: Input dataset
            
        Returns:
            Processed dataset
        """
        def process_batch(batch):
            # Load and process audio
            import librosa
            
            audio_arrays = []
            for audio_path in batch["audio"]:
                try:
                    audio, sr = librosa.load(audio_path, sr=self.sampling_rate)
                    audio_arrays.append(audio)
                except Exception as e:
                    logger.warning(f"Failed to load {audio_path}: {e}")
                    audio_arrays.append(np.zeros(self.sampling_rate))  # 1 second silence
            
            # Compute input features
            batch["input_features"] = self.processor(
                audio_arrays,
                sampling_rate=self.sampling_rate,
                return_tensors="np",
            ).input_features[0]
            
            # Process labels
            batch["labels"] = self.processor(
                text=batch["sentence"],
                return_tensors="np",
            ).input_ids[0]
            
            return batch
        
        dataset = dataset.map(
            process_batch,
            batched=True,
            batch_size=100,
            remove_columns=dataset.column_names,
        )
        
        return dataset


class DataCollatorSpeechSeq2Seq:
    """Data collator for speech seq2seq models."""
    
    def __init__(self, processor):
        """Initialize collator.
        
        Args:
            processor: HuggingFace processor
        """
        self.processor = processor
    
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        """Collate batch.
        
        Args:
            features: List of feature dicts
            
        Returns:
            Batched dict
        """
        # Split inputs and labels
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        
        # Pad input features
        batch = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt",
        )
        
        # Pad labels
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            return_tensors="pt",
        )
        
        # Replace padding with -100 for loss computation
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1),
            -100,
        )
        
        # Remove bos token if present
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        
        batch["labels"] = labels
        
        return batch


class FinetuneTrainer:
    """Fine-tune ASR models."""
    
    def __init__(self, config: FinetuneConfig):
        """Initialize trainer.
        
        Args:
            config: Fine-tuning configuration
        """
        self.config = config
        self.model = None
        self.processor = None
        self.trainer = None
    
    def setup(self) -> None:
        """Set up model and processor."""
        logger.info(f"Loading model: {self.config.model_name}")
        
        self.processor = AutoProcessor.from_pretrained(self.config.model_name)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16 if self.config.fp16 else torch.float32,
        )
        
        # Set language
        self.model.config.forced_decoder_ids = None
        self.model.config.suppress_tokens = []
        
        # Enable gradient checkpointing for memory efficiency
        self.model.gradient_checkpointing_enable()
        
        logger.info("Model loaded")
    
    def prepare_data(self) -> DatasetDict:
        """Prepare training and validation data.
        
        Returns:
            DatasetDict with train and validation sets
        """
        logger.info("Preparing datasets")
        
        dataset_prep = AudioDataset(
            processor=self.processor,
            max_input_length=self.config.max_input_length,
            max_label_length=self.config.max_label_length,
        )
        
        train_dataset = dataset_prep.load_manifest_to_dataset(self.config.train_manifest)
        val_dataset = dataset_prep.load_manifest_to_dataset(self.config.val_manifest)
        
        logger.info(f"Train samples: {len(train_dataset)}")
        logger.info(f"Val samples: {len(val_dataset)}")
        
        # Prepare datasets
        train_dataset = dataset_prep.prepare_dataset(train_dataset)
        val_dataset = dataset_prep.prepare_dataset(val_dataset)
        
        return DatasetDict({
            "train": train_dataset,
            "validation": val_dataset,
        })
    
    def compute_metrics(self, pred) -> Dict[str, float]:
        """Compute metrics during training.
        
        Args:
            pred: Prediction object
            
        Returns:
            Dict of metrics
        """
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        
        # Replace -100 with pad token id
        label_ids[label_ids == -100] = self.processor.tokenizer.pad_token_id
        
        # Decode
        pred_str = self.processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        
        # Compute WER
        import jiwer
        wer = jiwer.wer(label_str, pred_str) * 100
        
        return {"wer": wer}
    
    def train(self) -> None:
        """Run training."""
        if self.model is None:
            self.setup()
        
        datasets = self.prepare_data()
        
        # Data collator
        data_collator = DataCollatorSpeechSeq2Seq(self.processor)
        
        # Training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            num_train_epochs=self.config.num_epochs,
            eval_strategy="steps",
            eval_steps=self.config.eval_steps,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            gradient_checkpointing=True,
            fp16=self.config.fp16,
            dataloader_num_workers=self.config.dataloader_num_workers,
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            push_to_hub=False,
            report_to=["tensorboard"],
            seed=self.config.seed,
        )
        
        # Initialize trainer
        self.trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=datasets["train"],
            eval_dataset=datasets["validation"],
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            tokenizer=self.processor,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=self.config.early_stopping_patience)],
        )
        
        # Train
        logger.info("Starting training")
        self.trainer.train()
        
        # Save final model
        final_output_dir = Path(self.config.output_dir) / "final_model"
        self.trainer.save_model(final_output_dir)
        self.processor.save_pretrained(final_output_dir)
        
        logger.info(f"Training complete. Model saved to {final_output_dir}")
    
    def save_config(self) -> None:
        """Save training configuration."""
        config_path = Path(self.config.output_dir) / "finetune_config.json"
        with open(config_path, "w") as f:
            json.dump(self.config.__dict__, f, indent=2)


def run_finetune(
    model_name: str,
    train_manifest: Union[str, Path],
    val_manifest: Union[str, Path],
    output_dir: Union[str, Path],
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 1e-5,
) -> None:
    """Run fine-tuning from CLI.
    
    Args:
        model_name: Model name or path
        train_manifest: Training manifest
        val_manifest: Validation manifest
        output_dir: Output directory
        num_epochs: Number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
    """
    config = FinetuneConfig(
        model_name=model_name,
        output_dir=str(output_dir),
        train_manifest=str(train_manifest),
        val_manifest=str(val_manifest),
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    
    trainer = FinetuneTrainer(config)
    trainer.save_config()
    trainer.train()

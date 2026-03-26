"""Train/validation/test split creation module."""

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from tqdm import tqdm

from asr_pipeline.utils import get_logger, load_manifest, save_manifest

logger = get_logger(__name__)


class SplitCreator:
    """Create train/val/test splits with various strategies."""
    
    def __init__(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_seed: int = 42,
        stratify_by: Optional[str] = None,
        group_by: Optional[str] = None,
    ):
        """Initialize split creator.
        
        Args:
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing
            random_seed: Random seed
            stratify_by: Column to stratify by
            group_by: Column to group by (keeps groups together)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"
        
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed
        self.stratify_by = stratify_by
        self.group_by = group_by
    
    def create_random_split(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create random split.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with split column
        """
        np.random.seed(self.random_seed)
        
        n = len(df)
        indices = np.random.permutation(n)
        
        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
        
        df = df.copy()
        df["split"] = ""
        df.iloc[train_indices, df.columns.get_loc("split")] = "train"
        df.iloc[val_indices, df.columns.get_loc("split")] = "val"
        df.iloc[test_indices, df.columns.get_loc("split")] = "test"
        
        return df
    
    def create_stratified_split(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create stratified split.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with split column
        """
        if self.stratify_by not in df.columns:
            logger.warning(f"Stratify column {self.stratify_by} not found, using random split")
            return self.create_random_split(df)
        
        df = df.copy()
        df["split"] = ""
        
        # Create stratification labels
        y = df[self.stratify_by]
        
        # First split: train vs (val + test)
        val_test_ratio = self.val_ratio + self.test_ratio
        sss1 = StratifiedShuffleSplit(
            n_splits=1,
            test_size=val_test_ratio,
            random_state=self.random_seed,
        )
        
        for train_idx, val_test_idx in sss1.split(df, y):
            df.iloc[train_idx, df.columns.get_loc("split")] = "train"
            
            # Second split: val vs test
            val_test_df = df.iloc[val_test_idx]
            y_val_test = y.iloc[val_test_idx]
            
            test_ratio_of_val_test = self.test_ratio / val_test_ratio
            sss2 = StratifiedShuffleSplit(
                n_splits=1,
                test_size=test_ratio_of_val_test,
                random_state=self.random_seed,
            )
            
            for val_idx, test_idx in sss2.split(val_test_df, y_val_test):
                val_indices = val_test_df.iloc[val_idx].index
                test_indices = val_test_df.iloc[test_idx].index
                
                df.loc[val_indices, "split"] = "val"
                df.loc[test_indices, "split"] = "test"
        
        return df
    
    def create_group_split(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create group-based split (keeps groups together).
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with split column
        """
        if self.group_by not in df.columns:
            logger.warning(f"Group column {self.group_by} not found, using random split")
            return self.create_random_split(df)
        
        df = df.copy()
        df["split"] = ""
        
        groups = df[self.group_by].values
        
        # First split: train vs (val + test)
        val_test_ratio = self.val_ratio + self.test_ratio
        gss1 = GroupShuffleSplit(
            n_splits=1,
            test_size=val_test_ratio,
            random_state=self.random_seed,
        )
        
        for train_idx, val_test_idx in gss1.split(df, groups=groups):
            df.iloc[train_idx, df.columns.get_loc("split")] = "train"
            
            # Second split: val vs test
            val_test_df = df.iloc[val_test_idx]
            val_test_groups = val_test_df[self.group_by].values
            
            test_ratio_of_val_test = self.test_ratio / val_test_ratio
            gss2 = GroupShuffleSplit(
                n_splits=1,
                test_size=test_ratio_of_val_test,
                random_state=self.random_seed,
            )
            
            for val_idx, test_idx in gss2.split(val_test_df, groups=val_test_groups):
                val_indices = val_test_df.iloc[val_idx].index
                test_indices = val_test_df.iloc[test_idx].index
                
                df.loc[val_indices, "split"] = "val"
                df.loc[test_indices, "split"] = "test"
        
        return df
    
    def create_duration_balanced_split(
        self,
        df: pd.DataFrame,
        n_bins: int = 10,
    ) -> pd.DataFrame:
        """Create split balanced by audio duration.
        
        Args:
            df: Input DataFrame
            n_bins: Number of duration bins
            
        Returns:
            DataFrame with split column
        """
        if "duration_sec" not in df.columns:
            logger.warning("Duration column not found, using random split")
            return self.create_random_split(df)
        
        # Create duration bins
        df = df.copy()
        df["duration_bin"] = pd.qcut(
            df["duration_sec"],
            q=n_bins,
            labels=False,
            duplicates="drop",
        )
        
        # Use stratified split with duration bins
        original_stratify = self.stratify_by
        self.stratify_by = "duration_bin"
        
        result = self.create_stratified_split(df)
        result = result.drop(columns=["duration_bin"])
        
        self.stratify_by = original_stratify
        
        return result
    
    def create_split(
        self,
        manifest_path: Union[str, Path],
        output_dir: Union[str, Path],
        split_strategy: str = "random",
    ) -> Dict[str, pd.DataFrame]:
        """Create splits from manifest.
        
        Args:
            manifest_path: Input manifest path
            output_dir: Output directory for split manifests
            split_strategy: Split strategy (random, stratified, group, duration)
            
        Returns:
            Dict of split_name -> DataFrame
        """
        df = load_manifest(manifest_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Filter to valid samples only
        if "is_valid" in df.columns:
            df = df[df["is_valid"]].copy()
        
        logger.info(f"Creating {split_strategy} split for {len(df)} samples")
        
        # Create split
        if split_strategy == "random":
            df = self.create_random_split(df)
        elif split_strategy == "stratified":
            df = self.create_stratified_split(df)
        elif split_strategy == "group":
            df = self.create_group_split(df)
        elif split_strategy == "duration":
            df = self.create_duration_balanced_split(df)
        else:
            raise ValueError(f"Unknown split strategy: {split_strategy}")
        
        # Save full manifest with splits
        save_manifest(df, output_dir / "manifest_with_splits.parquet")
        
        # Save individual splits
        splits = {}
        for split_name in ["train", "val", "test"]:
            split_df = df[df["split"] == split_name].copy()
            splits[split_name] = split_df
            
            output_path = output_dir / f"{split_name}.parquet"
            save_manifest(split_df, output_path)
            
            # Print stats
            duration_hours = split_df["duration_sec"].sum() / 3600 if "duration_sec" in split_df.columns else 0
            logger.info(
                f"{split_name}: {len(split_df)} samples, "
                f"{duration_hours:.2f} hours"
            )
        
        return splits


def run_split(
    manifest_path: Union[str, Path],
    output_dir: Union[str, Path],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    split_strategy: str = "random",
    stratify_by: Optional[str] = None,
    group_by: Optional[str] = None,
    random_seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """Run split creation from CLI.
    
    Args:
        manifest_path: Input manifest path
        output_dir: Output directory
        train_ratio: Training ratio
        val_ratio: Validation ratio
        test_ratio: Test ratio
        split_strategy: Split strategy
        stratify_by: Column to stratify by
        group_by: Column to group by
        random_seed: Random seed
        
    Returns:
        Dict of split_name -> DataFrame
    """
    creator = SplitCreator(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=random_seed,
        stratify_by=stratify_by,
        group_by=group_by,
    )
    
    return creator.create_split(manifest_path, output_dir, split_strategy)

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class SteeringAngleAnalyzer:
    """Analyze and normalize steering angle distributions."""
    
    def __init__(self, steering_data: np.ndarray):
        self.steering_data = steering_data
        self.stats = {}
        
    def analyze_distribution(self) -> Dict:
        """Comprehensive steering angle distribution analysis."""
        data = self.steering_data
        
        self.stats = {
            'mean': np.mean(data),
            'std': np.std(data),
            'min': np.min(data),
            'max': np.max(data),
            'median': np.median(data),
            'q25': np.percentile(data, 25),
            'q75': np.percentile(data, 75),
            'zero_ratio': np.sum(data == 0) / len(data),
            'near_zero_ratio': np.sum(np.abs(data) < 0.01) / len(data),
            'range': np.max(data) - np.min(data)
        }
        
        return self.stats
    
    def plot_distribution(self, figsize=(15, 5)):
        """Visualize steering angle distribution."""
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Histogram
        axes[0].hist(self.steering_data, bins=50, alpha=0.7, color='skyblue')
        axes[0].set_title('Steering Angle Distribution')
        axes[0].set_xlabel('Steering Angle')
        axes[0].set_ylabel('Frequency')
        axes[0].axvline(0, color='red', linestyle='--', alpha=0.7)
        
        # Box plot
        axes[1].boxplot(self.steering_data)
        axes[1].set_title('Steering Angle Box Plot')
        axes[1].set_ylabel('Steering Angle')
        
        # Cumulative distribution
        sorted_data = np.sort(self.steering_data)
        cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        axes[2].plot(sorted_data, cumulative)
        axes[2].set_title('Cumulative Distribution')
        axes[2].set_xlabel('Steering Angle')
        axes[2].set_ylabel('Cumulative Probability')
        
        plt.tight_layout()
        plt.show()
        
        # Print statistics
        self.print_stats()
    
    def print_stats(self):
        """Print comprehensive statistics."""
        if not self.stats:
            self.analyze_distribution()
            
        print("\n📊 STEERING ANGLE STATISTICS")
        print("="*40)
        print(f"Mean: {self.stats['mean']:.6f}")
        print(f"Std:  {self.stats['std']:.6f}")
        print(f"Min:  {self.stats['min']:.6f}")
        print(f"Max:  {self.stats['max']:.6f}")
        print(f"Range: {self.stats['range']:.6f}")
        print(f"Median: {self.stats['median']:.6f}")
        print(f"Q25-Q75: [{self.stats['q25']:.6f}, {self.stats['q75']:.6f}]")
        print(f"Zero values: {self.stats['zero_ratio']:.2%}")
        print(f"Near-zero (±0.01): {self.stats['near_zero_ratio']:.2%}")
    
    def normalize_steering(self, method='minmax') -> np.ndarray:
        """
        Normalize steering angles using specified method.
        
        Args:
            method: 'minmax', 'zscore', or 'tanh'
        """
        data = self.steering_data
        
        if method == 'minmax':
            # Scale to [-1, 1]
            data_range = self.stats['max'] - self.stats['min']
            normalized = 2 * (data - self.stats['min']) / data_range - 1
            
        elif method == 'zscore':
            # Z-score normalization
            normalized = (data - self.stats['mean']) / self.stats['std']
            
        elif method == 'tanh':
            # Tanh normalization (preserves sign, bounds to [-1,1])
            normalized = np.tanh(data / self.stats['std'])
            
        else:
            raise ValueError("Method must be 'minmax', 'zscore', or 'tanh'")
        
        print(f"✓ Normalized using {method}: range [{np.min(normalized):.3f}, {np.max(normalized):.3f}]")
        return normalized

class BalancedSampler:
    """Create balanced sampling weights for regression."""
    
    def __init__(self, steering_angles: np.ndarray, num_bins: int = 20):
        self.steering_angles = steering_angles
        self.num_bins = num_bins
        self.bin_edges = None
        self.weights = None
        
    def create_binned_weights(self) -> np.ndarray:
        """Create sampling weights based on steering angle bins."""
        # Create bins
        self.bin_edges = np.linspace(
            np.min(self.steering_angles), 
            np.max(self.steering_angles), 
            self.num_bins + 1
        )
        
        # Assign samples to bins
        bin_indices = np.digitize(self.steering_angles, self.bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, self.num_bins - 1)
        
        # Count samples per bin
        bin_counts = np.bincount(bin_indices, minlength=self.num_bins)
        
        # Calculate inverse frequency weights
        bin_weights = 1.0 / (bin_counts + 1e-8)  # Add epsilon to avoid division by zero
        
        # Assign weights to samples
        sample_weights = bin_weights[bin_indices]
        
        # Normalize weights
        sample_weights = sample_weights / np.sum(sample_weights) * len(sample_weights)
        
        self.weights = sample_weights
        return sample_weights
    
    def create_magnitude_weights(self, power: float = 2.0) -> np.ndarray:
        """Create weights based on steering angle magnitude."""
        # Higher weights for larger steering angles
        magnitudes = np.abs(self.steering_angles)
        weights = np.power(magnitudes + 0.01, power)  # Add small constant
        
        # Normalize
        weights = weights / np.mean(weights)
        
        self.weights = weights
        return weights
    
    def get_sampler(self) -> WeightedRandomSampler:
        """Get PyTorch WeightedRandomSampler."""
        if self.weights is None:
            self.create_binned_weights()
            
        return WeightedRandomSampler(
            weights=torch.FloatTensor(self.weights),
            num_samples=len(self.weights),
            replacement=True
        )

class MultiViewDataset(Dataset):
    """Dataset for multi-view car steering prediction."""
    
    def __init__(
        self, 
        dataframe: pd.DataFrame,
        img_folder: str = 'IMG',
        transform: Optional[transforms.Compose] = None,
        steering_normalization: str = 'minmax'
    ):
        self.df = dataframe.reset_index(drop=True)
        self.img_folder = Path(img_folder)
        self.transform = transform
        
        # Analyze and normalize steering angles
        self.analyzer = SteeringAngleAnalyzer(self.df.iloc[:, 3].values)  # Assuming steering is 4th column
        self.analyzer.analyze_distribution()
        self.normalized_steering = self.analyzer.normalize_steering(steering_normalization)
        
        # Create balanced sampler
        self.sampler = BalancedSampler(self.normalized_steering)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load three images
        center_path = self.img_folder / Path(row.iloc[0]).name
        left_path = self.img_folder / Path(row.iloc[1]).name
        right_path = self.img_folder / Path(row.iloc[2]).name
        
        # Load images
        try:
            center_img = Image.open(center_path).convert('RGB')
            left_img = Image.open(left_path).convert('RGB')
            right_img = Image.open(right_path).convert('RGB')
        except Exception as e:
            # Fallback to zeros if image loading fails
            center_img = Image.new('RGB', (320, 160), color='black')
            left_img = Image.new('RGB', (320, 160), color='black')
            right_img = Image.new('RGB', (320, 160), color='black')
        
        # Apply transforms
        if self.transform:
            center_img = self.transform(center_img)
            left_img = self.transform(left_img)
            right_img = self.transform(right_img)
        
        # Get normalized steering angle
        steering = torch.FloatTensor([self.normalized_steering[idx]])
        
        # Get other features
        throttle = torch.FloatTensor([row.iloc[4]])  # Throttle
        brake = torch.FloatTensor([row.iloc[5]])     # Brake  
        speed = torch.FloatTensor([row.iloc[6]])     # Speed
        
        return {
            'center': center_img,
            'left': left_img, 
            'right': right_img,
            'steering': steering,
            'throttle': throttle,
            'brake': brake,
            'speed': speed
        }

class DataPreprocessor:
    """Main preprocessing pipeline."""
    
    def __init__(self, csv_path: str = 'driving_log.csv', img_folder: str = 'IMG'):
        self.csv_path = csv_path
        self.img_folder = img_folder
        self.data = None
        self.train_dataset = None
        self.val_dataset = None
        
    def load_data(self):
        """Load the driving log CSV."""
        self.data = pd.read_csv(self.csv_path)
        print(f"✓ Loaded {len(self.data)} samples")
        return self.data
    
    def get_transforms(self, is_training: bool = True) -> transforms.Compose:
        """Get image transforms for training or validation."""
        
        base_transforms = [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet stats
                std=[0.229, 0.224, 0.225]
            )
        ]
        
        if is_training:
            # Add augmentations for training
            augment_transforms = [
                transforms.Resize((224, 224)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.01)  # Gaussian noise
            ]
            return transforms.Compose(augment_transforms)
        
        return transforms.Compose(base_transforms)
    
    def create_train_val_split(
        self, 
        test_size: float = 0.2, 
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create train/validation split preserving temporal order if needed."""
        
        if self.data is None:
            self.load_data()
        
        # Simple random split (can be modified for temporal split)
        train_df, val_df = train_test_split(
            self.data, 
            test_size=test_size, 
            random_state=random_state,
            shuffle=True
        )
        
        print(f"✓ Train: {len(train_df)} samples, Val: {len(val_df)} samples")
        return train_df, val_df
    
    def create_datasets(
        self, 
        steering_normalization: str = 'minmax',
        test_size: float = 0.2
    ) -> Tuple[MultiViewDataset, MultiViewDataset]:
        """Create training and validation datasets."""
        
        train_df, val_df = self.create_train_val_split(test_size=test_size)
        
        # Create transforms
        train_transforms = self.get_transforms(is_training=True)
        val_transforms = self.get_transforms(is_training=False)
        
        # Create datasets
        self.train_dataset = MultiViewDataset(
            train_df, 
            self.img_folder, 
            train_transforms,
            steering_normalization
        )
        
        self.val_dataset = MultiViewDataset(
            val_df, 
            self.img_folder, 
            val_transforms,
            steering_normalization
        )
        
        return self.train_dataset, self.val_dataset
    
    def create_dataloaders(
        self, 
        batch_size: int = 32,
        use_balanced_sampling: bool = True,
        num_workers: int = 4
    ) -> Tuple[DataLoader, DataLoader]:
        """Create training and validation dataloaders."""
        
        if self.train_dataset is None:
            self.create_datasets()
        
        # Get balanced sampler for training
        train_sampler = None
        if use_balanced_sampling:
            train_sampler = self.train_dataset.sampler.get_sampler()
        
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        print(f"✓ Created dataloaders: batch_size={batch_size}, balanced_sampling={use_balanced_sampling}")
        return train_loader, val_loader
    
    def analyze_preprocessing(self):
        """Analyze preprocessing results."""
        if self.train_dataset is None:
            self.create_datasets()
        
        print("\n🔍 PREPROCESSING ANALYSIS")
        print("="*50)
        
        # Analyze training data steering distribution
        print("\n📊 Training Data Steering Analysis:")
        self.train_dataset.analyzer.print_stats()
        
        # Test data loading
        print("\n🧪 Testing data loading...")
        sample = self.train_dataset[0]
        
        print(f"✓ Image shapes: center={sample['center'].shape}, left={sample['left'].shape}, right={sample['right'].shape}")
        print(f"✓ Steering range: [{torch.min(sample['steering']):.3f}, {torch.max(sample['steering']):.3f}]")
        print(f"✓ Data types: steering={sample['steering'].dtype}, images={sample['center'].dtype}")
        
        # Test dataloader
        train_loader, _ = self.create_dataloaders(batch_size=2)
        batch = next(iter(train_loader))
        
        print(f"✓ Batch shapes: center={batch['center'].shape}, steering={batch['steering'].shape}")
        print("✅ Preprocessing pipeline ready!")

# Usage example
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = DataPreprocessor('path')
    
    # Create datasets and analyze
    train_dataset, val_dataset = preprocessor.create_datasets(
        steering_normalization='minmax',
        test_size=0.2
    )
    
    # Analyze preprocessing
    preprocessor.analyze_preprocessing()
    
    # Visualize steering distribution
    train_dataset.analyzer.plot_distribution()
    
    # Create dataloaders
    train_loader, val_loader = preprocessor.create_dataloaders(
        batch_size=32,
        use_balanced_sampling=True
    )
    
    print(f"\n🎯 Ready for training with {len(train_loader)} train batches and {len(val_loader)} val batches!")

"""
Unit tests for core modules.
"""

import pytest
import torch
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.core.base import BaseConfig, set_seed


class TestBaseConfig:
    """Test BaseConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = BaseConfig()
        assert config.seed == 42
        assert config.device in ["cpu", "cuda"]
        assert config.paths == {}
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = BaseConfig(seed=123, device="cpu", paths={"data": "/tmp"})
        assert config.seed == 123
        assert config.device == "cpu"
        assert config.paths == {"data": "/tmp"}
    
    def test_to_dict(self):
        """Test config to dictionary conversion."""
        config = BaseConfig(seed=123)
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict["seed"] == 123
    
    def test_save_and_load(self, tmp_path):
        """Test saving and loading configuration."""
        config = BaseConfig(seed=123, device="cpu")
        save_path = tmp_path / "config.json"
        
        config.save(save_path)
        loaded_config = BaseConfig.load(save_path)
        
        assert loaded_config.seed == 123
        assert loaded_config.device == "cpu"


class TestSetSeed:
    """Test set_seed function."""
    
    def test_set_seed_reproducibility(self):
        """Test that setting seed produces reproducible results."""
        set_seed(42)
        rand1 = np.random.rand()
        torch_rand1 = torch.rand(1).item()
        
        set_seed(42)
        rand2 = np.random.rand()
        torch_rand2 = torch.rand(1).item()
        
        assert rand1 == rand2
        assert torch_rand1 == torch_rand2

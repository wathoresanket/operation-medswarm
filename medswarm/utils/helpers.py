"""
Utility Functions
==================
Helper functions for the MedSwarm project.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import yaml


def get_project_root() -> Path:
    """Get the project root directory."""
    current = Path(__file__).resolve()
    # Navigate up to find the project root (contains config/ directory)
    for parent in current.parents:
        if (parent / "config").exists() or (parent / "medswarm").exists():
            return parent
    return current.parent.parent.parent


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file (default: config/config.yaml)
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = get_project_root() / "config" / "config.yaml"
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_logging(
    name: str = "medswarm",
    level: int = logging.INFO,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging for the project.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional file path for log output
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_format = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
        
    return logger


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def format_number(num: float) -> str:
    """Format large numbers with K/M suffixes."""
    if abs(num) >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    elif abs(num) >= 1_000:
        return f"{num/1_000:.1f}K"
    else:
        return f"{num:.1f}"

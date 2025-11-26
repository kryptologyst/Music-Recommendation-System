"""Music Recommendation System - Main Package."""

__version__ = "1.0.0"
__author__ = "AI Assistant"
__email__ = "ai@example.com"

from .data import MusicDataLoader, MusicDataset
from .models import (
    ContentBasedRecommender,
    CollaborativeFilteringRecommender,
    HybridRecommender,
)
from .evaluation import RecommenderEvaluator
from .utils import set_seed, load_config

__all__ = [
    "MusicDataLoader",
    "MusicDataset", 
    "ContentBasedRecommender",
    "CollaborativeFilteringRecommender",
    "HybridRecommender",
    "RecommenderEvaluator",
    "set_seed",
    "load_config",
]

#!/usr/bin/env python3
"""Main training script for music recommendation system."""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from src.data import MusicDataLoader
from src.models import ContentBasedRecommender, CollaborativeFilteringRecommender, HybridRecommender
from src.evaluation import RecommenderEvaluator
from src.utils import load_config, set_seed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train music recommendation models")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--save-models", 
        action="store_true",
        help="Save trained models"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)
    
    # Set random seed
    set_seed(config['data']['seed'])
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    logger.info("Loading dataset...")
    data_loader = MusicDataLoader(
        data_dir=config['data']['data_dir'],
        seed=config['data']['seed']
    )
    
    dataset = data_loader.load_data()
    
    # Create train/test split
    train_dataset, test_dataset = data_loader.create_train_test_split(
        dataset,
        test_size=config['data']['test_size'],
        time_based=config['data']['time_based_split']
    )
    
    logger.info(f"Dataset split: {len(train_dataset.interactions_df)} train, {len(test_dataset.interactions_df)} test interactions")
    
    # Initialize models
    models = {
        'Content-Based': ContentBasedRecommender(
            use_sentence_transformer=config['models']['content_based']['use_sentence_transformer'],
            seed=config['models']['content_based']['seed']
        ),
        'Collaborative Filtering': CollaborativeFilteringRecommender(
            model_type=config['models']['collaborative_filtering']['model_type'],
            factors=config['models']['collaborative_filtering']['factors'],
            regularization=config['models']['collaborative_filtering']['regularization'],
            iterations=config['models']['collaborative_filtering']['iterations'],
            seed=config['models']['collaborative_filtering']['seed']
        ),
        'Hybrid': HybridRecommender(
            content_weight=config['models']['hybrid']['content_weight'],
            collab_weight=config['models']['hybrid']['collab_weight'],
            seed=config['models']['hybrid']['seed']
        )
    }
    
    # Train models
    logger.info("Training models...")
    for name, model in models.items():
        logger.info(f"Training {name} model...")
        model.fit(train_dataset)
        logger.info(f"{name} model training completed")
    
    # Evaluate models
    logger.info("Evaluating models...")
    evaluator = RecommenderEvaluator(k_values=config['evaluation']['k_values'])
    
    results_df = evaluator.compare_models(models, test_dataset, train_dataset)
    
    # Print results
    evaluator.print_results(results_df)
    
    # Save results
    results_path = output_dir / "evaluation_results.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"Results saved to {results_path}")
    
    # Generate and save report
    report_path = output_dir / "evaluation_report.md"
    report = evaluator.generate_report(models, test_dataset, train_dataset, str(report_path))
    logger.info(f"Report saved to {report_path}")
    
    # Save models if requested
    if args.save_models:
        import pickle
        
        models_path = output_dir / "trained_models.pkl"
        with open(models_path, 'wb') as f:
            pickle.dump(models, f)
        logger.info(f"Models saved to {models_path}")
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()

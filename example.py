#!/usr/bin/env python3
"""Simple example script demonstrating the music recommendation system."""

from src.data import MusicDataLoader
from src.models import ContentBasedRecommender, CollaborativeFilteringRecommender, HybridRecommender
from src.evaluation import RecommenderEvaluator
from src.utils import set_seed

def main():
    """Run a simple example of the music recommendation system."""
    print("🎵 Music Recommendation System - Simple Example")
    print("=" * 50)
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Load data
    print("Loading music dataset...")
    data_loader = MusicDataLoader(data_dir="data", seed=42)
    dataset = data_loader.load_data()
    
    print(f"Dataset loaded: {len(dataset.user_to_idx)} users, {len(dataset.item_to_idx)} items")
    
    # Create train/test split
    train_dataset, test_dataset = data_loader.create_train_test_split(dataset, test_size=0.2)
    
    # Initialize and train models
    print("\nTraining models...")
    models = {
        'Content-Based': ContentBasedRecommender(use_sentence_transformer=True, seed=42),
        'Collaborative Filtering': CollaborativeFilteringRecommender(model_type='als', seed=42),
        'Hybrid': HybridRecommender(content_weight=0.3, collab_weight=0.7, seed=42)
    }
    
    for name, model in models.items():
        print(f"  Training {name}...")
        model.fit(train_dataset)
    
    # Get recommendations for a sample user
    sample_user = list(train_dataset.user_to_idx.keys())[0]
    print(f"\nGetting recommendations for user: {sample_user}")
    
    # Show user's history
    user_history = train_dataset.get_user_interactions(sample_user)
    print(f"\nUser's music history ({len(user_history)} songs):")
    for _, row in user_history.head(3).iterrows():
        item_info = train_dataset.get_item_features(row['item_id'])
        print(f"  - {item_info['title']} by {item_info['artist']} ({item_info['genre']})")
    
    # Get recommendations
    print(f"\nRecommendations:")
    for model_name, model in models.items():
        recommendations = model.recommend(sample_user, n_recommendations=3)
        print(f"\n{model_name}:")
        for i, (item_id, score) in enumerate(recommendations, 1):
            item_info = train_dataset.get_item_features(item_id)
            print(f"  {i}. {item_info['title']} by {item_info['artist']} ({item_info['genre']}) - Score: {score:.4f}")
    
    # Evaluate models
    print(f"\nEvaluating models...")
    evaluator = RecommenderEvaluator(k_values=[5, 10])
    results_df = evaluator.compare_models(models, test_dataset, train_dataset)
    
    print(f"\nModel Performance (NDCG@10):")
    for _, row in results_df.iterrows():
        print(f"  {row['model']}: {row['ndcg@10']:.4f}")
    
    print(f"\n✅ Example completed successfully!")
    print(f"Run 'streamlit run demo.py' for the interactive demo!")

if __name__ == "__main__":
    main()

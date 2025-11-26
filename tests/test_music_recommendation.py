"""Unit tests for music recommendation system."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from src.data import MusicDataset, MusicDataLoader
from src.models import ContentBasedRecommender, CollaborativeFilteringRecommender, HybridRecommender
from src.evaluation import RecommenderEvaluator
from src.utils import set_seed, load_config


class TestMusicDataset:
    """Test cases for MusicDataset class."""
    
    def setup_method(self):
        """Set up test data."""
        # Create sample data
        self.interactions_df = pd.DataFrame({
            'user_id': ['user1', 'user1', 'user2', 'user2'],
            'item_id': ['item1', 'item2', 'item1', 'item3'],
            'timestamp': pd.date_range('2023-01-01', periods=4),
            'weight': [1, 2, 1, 3]
        })
        
        self.items_df = pd.DataFrame({
            'item_id': ['item1', 'item2', 'item3'],
            'title': ['Song 1', 'Song 2', 'Song 3'],
            'genre': ['pop', 'rock', 'jazz'],
            'description': ['Pop song', 'Rock song', 'Jazz song']
        })
        
        self.users_df = pd.DataFrame({
            'user_id': ['user1', 'user2'],
            'age': [25, 30],
            'gender': ['M', 'F']
        })
        
        self.dataset = MusicDataset(self.interactions_df, self.items_df, self.users_df)
    
    def test_dataset_initialization(self):
        """Test dataset initialization."""
        assert len(self.dataset.user_to_idx) == 2
        assert len(self.dataset.item_to_idx) == 3
        assert len(self.dataset.interactions_df) == 4
    
    def test_get_user_interactions(self):
        """Test getting user interactions."""
        user_interactions = self.dataset.get_user_interactions('user1')
        assert len(user_interactions) == 2
        assert set(user_interactions['item_id']) == {'item1', 'item2'}
    
    def test_get_item_interactions(self):
        """Test getting item interactions."""
        item_interactions = self.dataset.get_item_interactions('item1')
        assert len(item_interactions) == 2
        assert set(item_interactions['user_id']) == {'user1', 'user2'}
    
    def test_get_item_features(self):
        """Test getting item features."""
        item_features = self.dataset.get_item_features('item1')
        assert item_features['title'] == 'Song 1'
        assert item_features['genre'] == 'pop'
    
    def test_get_user_features(self):
        """Test getting user features."""
        user_features = self.dataset.get_user_features('user1')
        assert user_features['age'] == 25
        assert user_features['gender'] == 'M'
    
    def test_invalid_data_validation(self):
        """Test data validation with invalid data."""
        invalid_interactions = pd.DataFrame({
            'user_id': ['user1'],
            'item_id': ['item1']
            # Missing timestamp and weight columns
        })
        
        with pytest.raises(ValueError):
            MusicDataset(invalid_interactions, self.items_df)


class TestMusicDataLoader:
    """Test cases for MusicDataLoader class."""
    
    def setup_method(self):
        """Set up test data loader."""
        self.data_loader = MusicDataLoader(data_dir="test_data", seed=42)
    
    @patch('src.data.Path.exists')
    def test_load_from_existing_files(self, mock_exists):
        """Test loading from existing files."""
        mock_exists.return_value = True
        
        with patch('pandas.read_csv') as mock_read_csv:
            mock_interactions = pd.DataFrame({
                'user_id': ['user1'],
                'item_id': ['item1'],
                'timestamp': [pd.Timestamp.now()],
                'weight': [1]
            })
            mock_items = pd.DataFrame({
                'item_id': ['item1'],
                'title': ['Song 1']
            })
            
            mock_read_csv.side_effect = [mock_interactions, mock_items]
            
            dataset = self.data_loader.load_data()
            assert len(dataset.interactions_df) == 1
    
    def test_generate_synthetic_data(self):
        """Test synthetic data generation."""
        with patch('src.data.Path.exists', return_value=False):
            dataset = self.data_loader._generate_synthetic_data()
            
            assert len(dataset.items_df) == 1000
            assert len(dataset.users_df) == 200
            assert len(dataset.interactions_df) > 0
    
    def test_create_train_test_split(self):
        """Test train/test split creation."""
        # Create a small dataset for testing
        interactions_df = pd.DataFrame({
            'user_id': ['user1'] * 10,
            'item_id': [f'item{i}' for i in range(10)],
            'timestamp': pd.date_range('2023-01-01', periods=10),
            'weight': [1] * 10
        })
        
        items_df = pd.DataFrame({
            'item_id': [f'item{i}' for i in range(10)],
            'title': [f'Song {i}' for i in range(10)]
        })
        
        dataset = MusicDataset(interactions_df, items_df)
        
        train_dataset, test_dataset = self.data_loader.create_train_test_split(
            dataset, test_size=0.2, time_based=True
        )
        
        assert len(train_dataset.interactions_df) == 8
        assert len(test_dataset.interactions_df) == 2


class TestContentBasedRecommender:
    """Test cases for ContentBasedRecommender class."""
    
    def setup_method(self):
        """Set up test data."""
        self.recommender = ContentBasedRecommender(use_sentence_transformer=False, seed=42)
        
        # Create sample dataset
        interactions_df = pd.DataFrame({
            'user_id': ['user1', 'user1', 'user2'],
            'item_id': ['item1', 'item2', 'item1'],
            'timestamp': pd.date_range('2023-01-01', periods=3),
            'weight': [1, 2, 1]
        })
        
        items_df = pd.DataFrame({
            'item_id': ['item1', 'item2', 'item3'],
            'title': ['Pop Song', 'Rock Song', 'Jazz Song'],
            'description': ['Upbeat pop song', 'Heavy rock song', 'Smooth jazz song']
        })
        
        self.dataset = MusicDataset(interactions_df, items_df)
    
    def test_fit_model(self):
        """Test model fitting."""
        self.recommender.fit(self.dataset)
        assert self.recommender.is_fitted
        assert len(self.recommender.user_profiles) == 2
    
    def test_recommend(self):
        """Test recommendation generation."""
        self.recommender.fit(self.dataset)
        
        recommendations = self.recommender.recommend('user1', n_recommendations=2)
        assert len(recommendations) <= 2
        assert all(isinstance(item_id, str) for item_id, _ in recommendations)
        assert all(isinstance(score, float) for _, score in recommendations)
    
    def test_cold_start_user(self):
        """Test recommendations for unknown user."""
        self.recommender.fit(self.dataset)
        
        recommendations = self.recommender.recommend('unknown_user', n_recommendations=2)
        assert len(recommendations) == 2
    
    def test_get_similar_items(self):
        """Test similar items retrieval."""
        self.recommender.fit(self.dataset)
        
        similar_items = self.recommender.get_similar_items('item1', n_similar=2)
        assert len(similar_items) <= 2
        assert all(isinstance(item_id, str) for item_id, _ in similar_items)


class TestCollaborativeFilteringRecommender:
    """Test cases for CollaborativeFilteringRecommender class."""
    
    def setup_method(self):
        """Set up test data."""
        self.recommender = CollaborativeFilteringRecommender(model_type='als', seed=42)
        
        # Create sample dataset
        interactions_df = pd.DataFrame({
            'user_id': ['user1', 'user1', 'user2', 'user2'],
            'item_id': ['item1', 'item2', 'item1', 'item3'],
            'timestamp': pd.date_range('2023-01-01', periods=4),
            'weight': [1, 2, 1, 3]
        })
        
        items_df = pd.DataFrame({
            'item_id': ['item1', 'item2', 'item3'],
            'title': ['Song 1', 'Song 2', 'Song 3']
        })
        
        self.dataset = MusicDataset(interactions_df, items_df)
    
    def test_fit_model(self):
        """Test model fitting."""
        self.recommender.fit(self.dataset)
        assert self.recommender.is_fitted
        assert self.recommender.user_item_matrix.shape == (2, 3)
    
    def test_recommend(self):
        """Test recommendation generation."""
        self.recommender.fit(self.dataset)
        
        recommendations = self.recommender.recommend('user1', n_recommendations=2)
        assert len(recommendations) <= 2
        assert all(isinstance(item_id, str) for item_id, _ in recommendations)


class TestHybridRecommender:
    """Test cases for HybridRecommender class."""
    
    def setup_method(self):
        """Set up test data."""
        self.recommender = HybridRecommender(seed=42)
        
        # Create sample dataset
        interactions_df = pd.DataFrame({
            'user_id': ['user1', 'user1', 'user2'],
            'item_id': ['item1', 'item2', 'item1'],
            'timestamp': pd.date_range('2023-01-01', periods=3),
            'weight': [1, 2, 1]
        })
        
        items_df = pd.DataFrame({
            'item_id': ['item1', 'item2', 'item3'],
            'title': ['Pop Song', 'Rock Song', 'Jazz Song'],
            'description': ['Upbeat pop song', 'Heavy rock song', 'Smooth jazz song']
        })
        
        self.dataset = MusicDataset(interactions_df, items_df)
    
    def test_fit_model(self):
        """Test model fitting."""
        self.recommender.fit(self.dataset)
        assert self.recommender.is_fitted
        assert self.recommender.content_model.is_fitted
        assert self.recommender.collab_model.is_fitted
    
    def test_recommend(self):
        """Test recommendation generation."""
        self.recommender.fit(self.dataset)
        
        recommendations = self.recommender.recommend('user1', n_recommendations=2)
        assert len(recommendations) <= 2
        assert all(isinstance(item_id, str) for item_id, _ in recommendations)


class TestRecommenderEvaluator:
    """Test cases for RecommenderEvaluator class."""
    
    def setup_method(self):
        """Set up test data."""
        self.evaluator = RecommenderEvaluator(k_values=[5, 10])
        
        # Create sample datasets
        interactions_df = pd.DataFrame({
            'user_id': ['user1', 'user1', 'user2', 'user2'],
            'item_id': ['item1', 'item2', 'item1', 'item3'],
            'timestamp': pd.date_range('2023-01-01', periods=4),
            'weight': [1, 2, 1, 3]
        })
        
        items_df = pd.DataFrame({
            'item_id': ['item1', 'item2', 'item3'],
            'title': ['Song 1', 'Song 2', 'Song 3']
        })
        
        self.train_dataset = MusicDataset(interactions_df, items_df)
        
        # Create test dataset with different interactions
        test_interactions_df = pd.DataFrame({
            'user_id': ['user1', 'user2'],
            'item_id': ['item3', 'item2'],
            'timestamp': pd.date_range('2023-02-01', periods=2),
            'weight': [1, 1]
        })
        
        self.test_dataset = MusicDataset(test_interactions_df, items_df)
    
    def test_evaluate_model(self):
        """Test model evaluation."""
        # Create a mock model
        mock_model = MagicMock()
        mock_model.recommend.return_value = [('item3', 0.8), ('item2', 0.6)]
        
        metrics = self.evaluator.evaluate_model(
            mock_model, self.test_dataset, self.train_dataset
        )
        
        assert 'precision@5' in metrics
        assert 'recall@5' in metrics
        assert 'hit_rate@5' in metrics
        assert 'ndcg@5' in metrics
    
    def test_compare_models(self):
        """Test model comparison."""
        # Create mock models
        mock_model1 = MagicMock()
        mock_model1.recommend.return_value = [('item3', 0.8)]
        
        mock_model2 = MagicMock()
        mock_model2.recommend.return_value = [('item2', 0.7)]
        
        models = {'Model1': mock_model1, 'Model2': mock_model2}
        
        results_df = self.evaluator.compare_models(models, self.test_dataset, self.train_dataset)
        
        assert len(results_df) == 2
        assert 'model' in results_df.columns


class TestUtils:
    """Test cases for utility functions."""
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        # This is hard to test directly, but we can ensure it doesn't raise an error
        assert True
    
    def test_load_config(self):
        """Test configuration loading."""
        # Create a temporary config file
        import tempfile
        import yaml
        
        config_data = {'test': 'value', 'nested': {'key': 123}}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
        
        try:
            loaded_config = load_config(temp_path)
            assert loaded_config == config_data
        finally:
            import os
            os.unlink(temp_path)
    
    def test_load_config_file_not_found(self):
        """Test configuration loading with non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_config('non_existent_file.yaml')


if __name__ == "__main__":
    pytest.main([__file__])

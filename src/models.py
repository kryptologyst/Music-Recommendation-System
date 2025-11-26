"""Recommendation models for music recommendation system."""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import implicit
from lightfm import LightFM
from lightfm.evaluation import precision_at_k, recall_at_k
from sentence_transformers import SentenceTransformer

from .data import MusicDataset
from .utils import set_seed

logger = logging.getLogger(__name__)


class BaseRecommender(ABC):
    """Base class for recommendation models."""
    
    def __init__(self, seed: int = 42):
        """Initialize base recommender.
        
        Args:
            seed: Random seed for reproducibility.
        """
        self.seed = seed
        set_seed(seed)
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, dataset: MusicDataset) -> None:
        """Fit the recommendation model.
        
        Args:
            dataset: Music dataset to train on.
        """
        pass
    
    @abstractmethod
    def recommend(
        self, 
        user_id: Union[str, int], 
        n_recommendations: int = 10,
        exclude_seen: bool = True
    ) -> List[Tuple[Union[str, int], float]]:
        """Generate recommendations for a user.
        
        Args:
            user_id: User ID to generate recommendations for.
            n_recommendations: Number of recommendations to generate.
            exclude_seen: Whether to exclude items the user has already interacted with.
            
        Returns:
            List of (item_id, score) tuples.
        """
        pass
    
    def get_similar_items(
        self, 
        item_id: Union[str, int], 
        n_similar: int = 10
    ) -> List[Tuple[Union[str, int], float]]:
        """Get items similar to a given item.
        
        Args:
            item_id: Item ID to find similar items for.
            n_similar: Number of similar items to return.
            
        Returns:
            List of (item_id, similarity_score) tuples.
        """
        raise NotImplementedError("Similar items not implemented for this model")


class ContentBasedRecommender(BaseRecommender):
    """Content-based recommendation using TF-IDF and cosine similarity."""
    
    def __init__(
        self, 
        use_sentence_transformer: bool = True,
        seed: int = 42
    ):
        """Initialize content-based recommender.
        
        Args:
            use_sentence_transformer: Whether to use sentence transformers for embeddings.
            seed: Random seed for reproducibility.
        """
        super().__init__(seed)
        self.use_sentence_transformer = use_sentence_transformer
        
        if use_sentence_transformer:
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        else:
            self.tfidf_vectorizer = TfidfVectorizer(
                stop_words='english',
                max_features=1000,
                ngram_range=(1, 2)
            )
        
        self.dataset = None
        self.item_features = None
        self.user_profiles = None
    
    def fit(self, dataset: MusicDataset) -> None:
        """Fit the content-based model.
        
        Args:
            dataset: Music dataset to train on.
        """
        self.dataset = dataset
        
        # Extract item features
        if self.use_sentence_transformer:
            descriptions = dataset.items_df['description'].fillna('').tolist()
            self.item_features = self.sentence_transformer.encode(descriptions)
        else:
            descriptions = dataset.items_df['description'].fillna('')
            self.item_features = self.tfidf_vectorizer.fit_transform(descriptions).toarray()
        
        # Build user profiles based on their interactions
        self.user_profiles = {}
        
        for user_id in dataset.user_to_idx.keys():
            user_interactions = dataset.get_user_interactions(user_id)
            if len(user_interactions) == 0:
                continue
            
            # Get items the user has interacted with
            interacted_items = user_interactions['item_id'].tolist()
            item_indices = [dataset.item_to_idx[item_id] for item_id in interacted_items]
            
            # Calculate weighted average of item features
            weights = user_interactions['weight'].values
            user_profile = np.average(
                self.item_features[item_indices], 
                axis=0, 
                weights=weights
            )
            
            self.user_profiles[user_id] = user_profile
        
        self.is_fitted = True
        logger.info(f"Content-based model fitted on {len(self.user_profiles)} users")
    
    def recommend(
        self, 
        user_id: Union[str, int], 
        n_recommendations: int = 10,
        exclude_seen: bool = True
    ) -> List[Tuple[Union[str, int], float]]:
        """Generate content-based recommendations.
        
        Args:
            user_id: User ID to generate recommendations for.
            n_recommendations: Number of recommendations to generate.
            exclude_seen: Whether to exclude items the user has already interacted with.
            
        Returns:
            List of (item_id, score) tuples.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        if user_id not in self.user_profiles:
            # Cold start: return popular items
            return self._get_popular_items(n_recommendations)
        
        user_profile = self.user_profiles[user_id]
        
        # Calculate similarity between user profile and all items
        similarities = cosine_similarity([user_profile], self.item_features)[0]
        
        # Get item IDs and scores
        item_scores = list(zip(self.dataset.idx_to_item.keys(), similarities))
        
        if exclude_seen:
            # Exclude items the user has already interacted with
            seen_items = set(self.dataset.get_user_interactions(user_id)['item_id'].tolist())
            item_scores = [(item_id, score) for item_id, score in item_scores 
                          if self.dataset.idx_to_item[item_id] not in seen_items]
        
        # Sort by similarity score and return top recommendations
        item_scores.sort(key=lambda x: x[1], reverse=True)
        return item_scores[:n_recommendations]
    
    def get_similar_items(
        self, 
        item_id: Union[str, int], 
        n_similar: int = 10
    ) -> List[Tuple[Union[str, int], float]]:
        """Get items similar to a given item based on content features."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before finding similar items")
        
        item_idx = self.dataset.item_to_idx[item_id]
        item_features = self.item_features[item_idx]
        
        # Calculate similarity with all other items
        similarities = cosine_similarity([item_features], self.item_features)[0]
        
        # Get item IDs and scores (excluding the item itself)
        item_scores = [(idx, score) for idx, score in enumerate(similarities) if idx != item_idx]
        item_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Convert indices to item IDs
        similar_items = [(self.dataset.idx_to_item[idx], score) for idx, score in item_scores[:n_similar]]
        return similar_items
    
    def _get_popular_items(self, n_items: int) -> List[Tuple[Union[str, int], float]]:
        """Get popular items for cold start users."""
        # Calculate popularity based on interaction count
        item_popularity = self.dataset.interactions_df.groupby('item_id')['weight'].sum().sort_values(ascending=False)
        popular_items = item_popularity.head(n_items)
        return [(item_id, float(score)) for item_id, score in popular_items.items()]


class CollaborativeFilteringRecommender(BaseRecommender):
    """Collaborative filtering using matrix factorization."""
    
    def __init__(
        self, 
        model_type: str = 'als',
        factors: int = 50,
        regularization: float = 0.01,
        iterations: int = 50,
        seed: int = 42
    ):
        """Initialize collaborative filtering recommender.
        
        Args:
            model_type: Type of model ('als', 'bpr', 'lightfm').
            factors: Number of latent factors.
            regularization: Regularization parameter.
            iterations: Number of training iterations.
            seed: Random seed for reproducibility.
        """
        super().__init__(seed)
        self.model_type = model_type
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        
        if model_type == 'lightfm':
            self.model = LightFM(
                no_components=factors,
                learning_rate=0.05,
                loss='warp',
                random_state=seed
            )
        else:
            # Use implicit library for ALS/BPR
            if model_type == 'als':
                self.model = implicit.als.AlternatingLeastSquares(
                    factors=factors,
                    regularization=regularization,
                    iterations=iterations,
                    random_state=seed
                )
            elif model_type == 'bpr':
                self.model = implicit.bpr.BayesianPersonalizedRanking(
                    factors=factors,
                    regularization=regularization,
                    iterations=iterations,
                    random_state=seed
                )
            else:
                raise ValueError(f"Unknown model type: {model_type}")
        
        self.dataset = None
        self.user_item_matrix = None
    
    def fit(self, dataset: MusicDataset) -> None:
        """Fit the collaborative filtering model.
        
        Args:
            dataset: Music dataset to train on.
        """
        self.dataset = dataset
        
        # Create user-item interaction matrix
        n_users = len(dataset.user_to_idx)
        n_items = len(dataset.item_to_idx)
        
        self.user_item_matrix = np.zeros((n_users, n_items))
        
        for _, row in dataset.interactions_df.iterrows():
            user_idx = dataset.user_to_idx[row['user_id']]
            item_idx = dataset.item_to_idx[row['item_id']]
            self.user_item_matrix[user_idx, item_idx] = row['weight']
        
        # Train model
        if self.model_type == 'lightfm':
            # LightFM expects sparse matrix
            from scipy.sparse import csr_matrix
            sparse_matrix = csr_matrix(self.user_item_matrix)
            self.model.fit(sparse_matrix)
        else:
            # Implicit library expects sparse matrix
            from scipy.sparse import csr_matrix
            sparse_matrix = csr_matrix(self.user_item_matrix)
            self.model.fit(sparse_matrix)
        
        self.is_fitted = True
        logger.info(f"Collaborative filtering model ({self.model_type}) fitted")
    
    def recommend(
        self, 
        user_id: Union[str, int], 
        n_recommendations: int = 10,
        exclude_seen: bool = True
    ) -> List[Tuple[Union[str, int], float]]:
        """Generate collaborative filtering recommendations.
        
        Args:
            user_id: User ID to generate recommendations for.
            n_recommendations: Number of recommendations to generate.
            exclude_seen: Whether to exclude items the user has already interacted with.
            
        Returns:
            List of (item_id, score) tuples.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        if user_id not in self.dataset.user_to_idx:
            # Cold start: return popular items
            return self._get_popular_items(n_recommendations)
        
        user_idx = self.dataset.user_to_idx[user_id]
        
        if self.model_type == 'lightfm':
            # LightFM recommendations
            user_items = self.user_item_matrix[user_idx].indices
            scores = self.model.predict(user_idx, np.arange(len(self.dataset.item_to_idx)))
            
            # Create item-score pairs
            item_scores = [(idx, score) for idx, score in enumerate(scores)]
            
            if exclude_seen:
                seen_items = set(self.dataset.get_user_interactions(user_id)['item_id'].tolist())
                item_scores = [(idx, score) for idx, score in item_scores 
                              if self.dataset.idx_to_item[idx] not in seen_items]
            
            item_scores.sort(key=lambda x: x[1], reverse=True)
            recommendations = [(self.dataset.idx_to_item[idx], float(score)) 
                             for idx, score in item_scores[:n_recommendations]]
        else:
            # Implicit library recommendations
            recommendations = self.model.recommend(
                user_idx, 
                self.user_item_matrix, 
                N=n_recommendations,
                filter_already_liked_items=exclude_seen
            )
            
            # Convert to (item_id, score) format
            recommendations = [(self.dataset.idx_to_item[item_idx], float(score)) 
                              for item_idx, score in recommendations]
        
        return recommendations
    
    def _get_popular_items(self, n_items: int) -> List[Tuple[Union[str, int], float]]:
        """Get popular items for cold start users."""
        item_popularity = self.dataset.interactions_df.groupby('item_id')['weight'].sum().sort_values(ascending=False)
        popular_items = item_popularity.head(n_items)
        return [(item_id, float(score)) for item_id, score in popular_items.items()]


class HybridRecommender(BaseRecommender):
    """Hybrid recommendation combining content-based and collaborative filtering."""
    
    def __init__(
        self,
        content_weight: float = 0.3,
        collab_weight: float = 0.7,
        seed: int = 42
    ):
        """Initialize hybrid recommender.
        
        Args:
            content_weight: Weight for content-based recommendations.
            collab_weight: Weight for collaborative filtering recommendations.
            seed: Random seed for reproducibility.
        """
        super().__init__(seed)
        self.content_weight = content_weight
        self.collab_weight = collab_weight
        
        self.content_model = ContentBasedRecommender(seed=seed)
        self.collab_model = CollaborativeFilteringRecommender(seed=seed)
        
        self.dataset = None
    
    def fit(self, dataset: MusicDataset) -> None:
        """Fit both content-based and collaborative filtering models.
        
        Args:
            dataset: Music dataset to train on.
        """
        self.dataset = dataset
        
        # Fit both models
        self.content_model.fit(dataset)
        self.collab_model.fit(dataset)
        
        self.is_fitted = True
        logger.info("Hybrid model fitted")
    
    def recommend(
        self, 
        user_id: Union[str, int], 
        n_recommendations: int = 10,
        exclude_seen: bool = True
    ) -> List[Tuple[Union[str, int], float]]:
        """Generate hybrid recommendations.
        
        Args:
            user_id: User ID to generate recommendations for.
            n_recommendations: Number of recommendations to generate.
            exclude_seen: Whether to exclude items the user has already interacted with.
            
        Returns:
            List of (item_id, score) tuples.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making recommendations")
        
        # Get recommendations from both models
        content_recs = self.content_model.recommend(user_id, n_recommendations * 2, exclude_seen)
        collab_recs = self.collab_model.recommend(user_id, n_recommendations * 2, exclude_seen)
        
        # Combine scores
        combined_scores = {}
        
        # Add content-based scores
        for item_id, score in content_recs:
            combined_scores[item_id] = combined_scores.get(item_id, 0) + self.content_weight * score
        
        # Add collaborative filtering scores
        for item_id, score in collab_recs:
            combined_scores[item_id] = combined_scores.get(item_id, 0) + self.collab_weight * score
        
        # Sort by combined score and return top recommendations
        sorted_items = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:n_recommendations]
    
    def get_similar_items(
        self, 
        item_id: Union[str, int], 
        n_similar: int = 10
    ) -> List[Tuple[Union[str, int], float]]:
        """Get similar items using content-based similarity."""
        return self.content_model.get_similar_items(item_id, n_similar)

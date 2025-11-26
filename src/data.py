"""Data loading and preprocessing for music recommendation system."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer

from .utils import set_seed

logger = logging.getLogger(__name__)


class MusicDataset:
    """Music dataset class for handling music data."""
    
    def __init__(
        self,
        interactions_df: pd.DataFrame,
        items_df: pd.DataFrame,
        users_df: Optional[pd.DataFrame] = None,
    ):
        """Initialize music dataset.
        
        Args:
            interactions_df: DataFrame with columns [user_id, item_id, timestamp, weight]
            items_df: DataFrame with item metadata
            users_df: Optional DataFrame with user metadata
        """
        self.interactions_df = interactions_df
        self.items_df = items_df
        self.users_df = users_df
        
        # Validate data
        self._validate_data()
        
        # Create mappings
        self.user_to_idx = {user: idx for idx, user in enumerate(self.interactions_df['user_id'].unique())}
        self.idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}
        self.item_to_idx = {item: idx for idx, item in enumerate(self.interactions_df['item_id'].unique())}
        self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}
        
        logger.info(f"Dataset loaded: {len(self.user_to_idx)} users, {len(self.item_to_idx)} items, "
                   f"{len(self.interactions_df)} interactions")
    
    def _validate_data(self) -> None:
        """Validate dataset format."""
        required_interaction_cols = ['user_id', 'item_id', 'timestamp', 'weight']
        required_item_cols = ['item_id', 'title']
        
        if not all(col in self.interactions_df.columns for col in required_interaction_cols):
            raise ValueError(f"Interactions DataFrame must contain columns: {required_interaction_cols}")
        
        if not all(col in self.items_df.columns for col in required_item_cols):
            raise ValueError(f"Items DataFrame must contain columns: {required_item_cols}")
    
    def get_user_interactions(self, user_id: Union[str, int]) -> pd.DataFrame:
        """Get interactions for a specific user."""
        return self.interactions_df[self.interactions_df['user_id'] == user_id]
    
    def get_item_interactions(self, item_id: Union[str, int]) -> pd.DataFrame:
        """Get interactions for a specific item."""
        return self.interactions_df[self.interactions_df['item_id'] == item_id]
    
    def get_item_features(self, item_id: Union[str, int]) -> pd.Series:
        """Get features for a specific item."""
        return self.items_df[self.items_df['item_id'] == item_id].iloc[0]
    
    def get_user_features(self, user_id: Union[str, int]) -> Optional[pd.Series]:
        """Get features for a specific user."""
        if self.users_df is None:
            return None
        user_data = self.users_df[self.users_df['user_id'] == user_id]
        return user_data.iloc[0] if len(user_data) > 0 else None


class MusicDataLoader:
    """Data loader for music recommendation system."""
    
    def __init__(self, data_dir: str = "data", seed: int = 42):
        """Initialize data loader.
        
        Args:
            data_dir: Directory containing data files.
            seed: Random seed for reproducibility.
        """
        self.data_dir = Path(data_dir)
        self.seed = seed
        set_seed(seed)
        
        # Initialize text encoders
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=1000,
            ngram_range=(1, 2)
        )
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
    
    def load_data(self) -> MusicDataset:
        """Load music dataset from files or generate synthetic data."""
        data_files = {
            'interactions': self.data_dir / 'interactions.csv',
            'items': self.data_dir / 'items.csv',
            'users': self.data_dir / 'users.csv'
        }
        
        # Check if data files exist
        if all(f.exists() for f in data_files.values()):
            logger.info("Loading existing data files...")
            return self._load_from_files(data_files)
        else:
            logger.info("Data files not found. Generating synthetic data...")
            return self._generate_synthetic_data()
    
    def _load_from_files(self, data_files: Dict[str, Path]) -> MusicDataset:
        """Load data from existing CSV files."""
        interactions_df = pd.read_csv(data_files['interactions'])
        items_df = pd.read_csv(data_files['items'])
        
        users_df = None
        if data_files['users'].exists():
            users_df = pd.read_csv(data_files['users'])
        
        return MusicDataset(interactions_df, items_df, users_df)
    
    def _generate_synthetic_data(self) -> MusicDataset:
        """Generate synthetic music data for demonstration."""
        np.random.seed(self.seed)
        
        # Generate items (songs)
        genres = ['pop', 'rock', 'electronic', 'classical', 'jazz', 'hip-hop', 'country', 'blues']
        moods = ['upbeat', 'calm', 'energetic', 'melancholic', 'romantic', 'aggressive', 'peaceful', 'dramatic']
        
        n_items = 1000
        items_data = []
        
        for i in range(n_items):
            genre = np.random.choice(genres)
            mood = np.random.choice(moods)
            tempo = np.random.randint(60, 180)
            
            # Create realistic song descriptions
            description = f"{genre} song with {mood} mood, tempo {tempo} BPM. "
            description += f"Features {np.random.choice(['catchy melody', 'complex harmonies', 'driving rhythm', 'emotional lyrics'])}. "
            description += f"Perfect for {np.random.choice(['workout', 'relaxation', 'party', 'study', 'driving', 'cooking'])}."
            
            items_data.append({
                'item_id': f'song_{i:04d}',
                'title': f"{genre.title()} Song {i+1}",
                'genre': genre,
                'mood': mood,
                'tempo': tempo,
                'description': description,
                'artist': f"Artist {i % 50 + 1}",
                'year': np.random.randint(1990, 2024),
                'duration': np.random.randint(120, 300),  # seconds
                'popularity': np.random.exponential(1.0)
            })
        
        items_df = pd.DataFrame(items_data)
        
        # Generate users
        n_users = 200
        users_data = []
        
        for i in range(n_users):
            age = np.random.randint(16, 65)
            gender = np.random.choice(['M', 'F', 'Other'])
            preferred_genres = np.random.choice(genres, size=np.random.randint(2, 5), replace=False)
            
            users_data.append({
                'user_id': f'user_{i:04d}',
                'age': age,
                'gender': gender,
                'preferred_genres': '|'.join(preferred_genres),
                'location': np.random.choice(['US', 'UK', 'CA', 'AU', 'DE', 'FR', 'JP']),
                'subscription_type': np.random.choice(['free', 'premium', 'family'])
            })
        
        users_df = pd.DataFrame(users_data)
        
        # Generate interactions (user-item interactions)
        interactions_data = []
        
        # Create realistic interaction patterns
        for user_idx, user_row in users_df.iterrows():
            user_id = user_row['user_id']
            preferred_genres = user_row['preferred_genres'].split('|')
            
            # Users interact with more items from preferred genres
            genre_items = items_df[items_df['genre'].isin(preferred_genres)]
            other_items = items_df[~items_df['genre'].isin(preferred_genres)]
            
            # Number of interactions per user (power law distribution)
            n_interactions = int(np.random.pareto(1.2) + 5)
            n_interactions = min(n_interactions, 100)  # Cap at 100
            
            # 70% interactions with preferred genres, 30% with others
            n_preferred = int(n_interactions * 0.7)
            n_other = n_interactions - n_preferred
            
            # Sample items
            if len(genre_items) > 0:
                preferred_item_ids = np.random.choice(genre_items['item_id'], 
                                                    size=min(n_preferred, len(genre_items)), 
                                                    replace=False)
            else:
                preferred_item_ids = []
            
            if len(other_items) > 0:
                other_item_ids = np.random.choice(other_items['item_id'], 
                                                size=min(n_other, len(other_items)), 
                                                replace=False)
            else:
                other_item_ids = []
            
            all_item_ids = list(preferred_item_ids) + list(other_item_ids)
            
            # Generate timestamps (last 2 years)
            base_time = pd.Timestamp.now() - pd.Timedelta(days=730)
            
            for item_id in all_item_ids:
                # Generate timestamp with recency bias
                days_ago = np.random.exponential(30)  # Most interactions in last 30 days
                timestamp = base_time + pd.Timedelta(days=days_ago)
                
                # Generate interaction weight (play count, rating, etc.)
                weight = np.random.poisson(3) + 1  # At least 1 interaction
                
                interactions_data.append({
                    'user_id': user_id,
                    'item_id': item_id,
                    'timestamp': timestamp,
                    'weight': weight
                })
        
        interactions_df = pd.DataFrame(interactions_data)
        
        # Save generated data
        self.data_dir.mkdir(exist_ok=True)
        interactions_df.to_csv(self.data_dir / 'interactions.csv', index=False)
        items_df.to_csv(self.data_dir / 'items.csv', index=False)
        users_df.to_csv(self.data_dir / 'users.csv', index=False)
        
        logger.info(f"Generated synthetic data: {len(users_df)} users, {len(items_df)} items, "
                   f"{len(interactions_df)} interactions")
        
        return MusicDataset(interactions_df, items_df, users_df)
    
    def create_train_test_split(
        self, 
        dataset: MusicDataset, 
        test_size: float = 0.2,
        time_based: bool = True
    ) -> Tuple[MusicDataset, MusicDataset]:
        """Create train/test split of the dataset.
        
        Args:
            dataset: Music dataset to split.
            test_size: Proportion of data to use for testing.
            time_based: If True, use time-based split. If False, use random split.
            
        Returns:
            Tuple of (train_dataset, test_dataset).
        """
        if time_based:
            # Time-based split: use last interactions for test
            interactions_df = dataset.interactions_df.copy()
            interactions_df = interactions_df.sort_values('timestamp')
            
            split_idx = int(len(interactions_df) * (1 - test_size))
            train_interactions = interactions_df.iloc[:split_idx]
            test_interactions = interactions_df.iloc[split_idx:]
        else:
            # Random split
            train_interactions, test_interactions = train_test_split(
                dataset.interactions_df, 
                test_size=test_size, 
                random_state=self.seed
            )
        
        train_dataset = MusicDataset(train_interactions, dataset.items_df, dataset.users_df)
        test_dataset = MusicDataset(test_interactions, dataset.items_df, dataset.users_df)
        
        return train_dataset, test_dataset
    
    def get_item_text_features(self, items_df: pd.DataFrame) -> np.ndarray:
        """Extract text features from item descriptions using TF-IDF."""
        descriptions = items_df['description'].fillna('')
        tfidf_features = self.tfidf_vectorizer.fit_transform(descriptions)
        return tfidf_features.toarray()
    
    def get_item_embeddings(self, items_df: pd.DataFrame) -> np.ndarray:
        """Extract semantic embeddings from item descriptions using sentence transformers."""
        descriptions = items_df['description'].fillna('').tolist()
        embeddings = self.sentence_transformer.encode(descriptions)
        return embeddings

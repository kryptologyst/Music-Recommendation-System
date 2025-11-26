"""Evaluation metrics and model comparison for music recommendation system."""

import logging
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, ndcg_score

from .data import MusicDataset
from .models import BaseRecommender

logger = logging.getLogger(__name__)


class RecommenderEvaluator:
    """Evaluator for recommendation models."""
    
    def __init__(self, k_values: List[int] = None):
        """Initialize evaluator.
        
        Args:
            k_values: List of k values for evaluation metrics.
        """
        self.k_values = k_values or [5, 10, 20]
    
    def evaluate_model(
        self,
        model: BaseRecommender,
        test_dataset: MusicDataset,
        train_dataset: MusicDataset,
        n_recommendations: int = 20
    ) -> Dict[str, float]:
        """Evaluate a recommendation model.
        
        Args:
            model: Trained recommendation model.
            test_dataset: Test dataset for evaluation.
            train_dataset: Training dataset (for excluding seen items).
            n_recommendations: Number of recommendations to generate per user.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        metrics = {}
        
        # Get all users in test set
        test_users = test_dataset.user_to_idx.keys()
        
        # Calculate metrics for each k
        for k in self.k_values:
            precision_scores = []
            recall_scores = []
            ndcg_scores = []
            hit_rates = []
            
            for user_id in test_users:
                # Get ground truth items for this user
                user_test_items = set(test_dataset.get_user_interactions(user_id)['item_id'].tolist())
                
                if len(user_test_items) == 0:
                    continue
                
                # Get recommendations
                try:
                    recommendations = model.recommend(
                        user_id, 
                        n_recommendations=n_recommendations,
                        exclude_seen=True
                    )
                    recommended_items = [item_id for item_id, _ in recommendations[:k]]
                except Exception as e:
                    logger.warning(f"Failed to get recommendations for user {user_id}: {e}")
                    continue
                
                # Calculate metrics
                if len(recommended_items) > 0:
                    # Precision@k
                    precision = len(set(recommended_items) & user_test_items) / len(recommended_items)
                    precision_scores.append(precision)
                    
                    # Recall@k
                    recall = len(set(recommended_items) & user_test_items) / len(user_test_items)
                    recall_scores.append(recall)
                    
                    # Hit Rate@k
                    hit_rate = 1.0 if len(set(recommended_items) & user_test_items) > 0 else 0.0
                    hit_rates.append(hit_rate)
                    
                    # NDCG@k
                    relevance_scores = [1.0 if item in user_test_items else 0.0 for item in recommended_items]
                    if sum(relevance_scores) > 0:
                        ndcg = ndcg_score([relevance_scores], [relevance_scores], k=k)
                        ndcg_scores.append(ndcg)
                    else:
                        ndcg_scores.append(0.0)
            
            # Average metrics
            metrics[f'precision@{k}'] = np.mean(precision_scores) if precision_scores else 0.0
            metrics[f'recall@{k}'] = np.mean(recall_scores) if recall_scores else 0.0
            metrics[f'hit_rate@{k}'] = np.mean(hit_rates) if hit_rates else 0.0
            metrics[f'ndcg@{k}'] = np.mean(ndcg_scores) if ndcg_scores else 0.0
        
        # Calculate additional metrics
        metrics.update(self._calculate_coverage_metrics(model, test_dataset, train_dataset))
        metrics.update(self._calculate_diversity_metrics(model, test_dataset))
        
        return metrics
    
    def _calculate_coverage_metrics(
        self,
        model: BaseRecommender,
        test_dataset: MusicDataset,
        train_dataset: MusicDataset
    ) -> Dict[str, float]:
        """Calculate coverage metrics."""
        all_items = set(train_dataset.item_to_idx.keys())
        recommended_items = set()
        
        # Get recommendations for all users
        for user_id in test_dataset.user_to_idx.keys():
            try:
                recommendations = model.recommend(user_id, n_recommendations=20, exclude_seen=True)
                user_recommended = {item_id for item_id, _ in recommendations}
                recommended_items.update(user_recommended)
            except Exception:
                continue
        
        # Catalog coverage
        catalog_coverage = len(recommended_items) / len(all_items) if all_items else 0.0
        
        return {
            'catalog_coverage': catalog_coverage,
            'unique_items_recommended': len(recommended_items)
        }
    
    def _calculate_diversity_metrics(
        self,
        model: BaseRecommender,
        test_dataset: MusicDataset
    ) -> Dict[str, float]:
        """Calculate diversity metrics."""
        if not hasattr(model, 'get_similar_items'):
            return {'intra_list_diversity': 0.0}
        
        diversity_scores = []
        
        # Sample users for diversity calculation
        sample_users = list(test_dataset.user_to_idx.keys())[:50]  # Limit for performance
        
        for user_id in sample_users:
            try:
                recommendations = model.recommend(user_id, n_recommendations=10, exclude_seen=True)
                if len(recommendations) < 2:
                    continue
                
                # Calculate intra-list diversity using item similarity
                similarities = []
                for i, (item1, _) in enumerate(recommendations):
                    for j, (item2, _) in enumerate(recommendations):
                        if i < j:
                            try:
                                similar_items = model.get_similar_items(item1, n_similar=100)
                                similarity = next((score for item, score in similar_items if item == item2), 0.0)
                                similarities.append(similarity)
                            except Exception:
                                similarities.append(0.0)
                
                if similarities:
                    avg_similarity = np.mean(similarities)
                    diversity = 1.0 - avg_similarity  # Higher diversity = lower similarity
                    diversity_scores.append(diversity)
            except Exception:
                continue
        
        return {
            'intra_list_diversity': np.mean(diversity_scores) if diversity_scores else 0.0
        }
    
    def compare_models(
        self,
        models: Dict[str, BaseRecommender],
        test_dataset: MusicDataset,
        train_dataset: MusicDataset
    ) -> pd.DataFrame:
        """Compare multiple models and return results as DataFrame.
        
        Args:
            models: Dictionary of model names to model instances.
            test_dataset: Test dataset for evaluation.
            train_dataset: Training dataset.
            
        Returns:
            DataFrame with evaluation results.
        """
        results = []
        
        for model_name, model in models.items():
            logger.info(f"Evaluating {model_name}...")
            metrics = self.evaluate_model(model, test_dataset, train_dataset)
            metrics['model'] = model_name
            results.append(metrics)
        
        results_df = pd.DataFrame(results)
        
        # Sort by NDCG@10 (or first available metric)
        sort_metric = None
        for k in self.k_values:
            if f'ndcg@{k}' in results_df.columns:
                sort_metric = f'ndcg@{k}'
                break
        
        if sort_metric:
            results_df = results_df.sort_values(sort_metric, ascending=False)
        
        return results_df
    
    def print_results(self, results_df: pd.DataFrame) -> None:
        """Print evaluation results in a formatted table."""
        print("\n" + "="*80)
        print("RECOMMENDATION MODEL EVALUATION RESULTS")
        print("="*80)
        
        # Round numeric columns
        numeric_cols = results_df.select_dtypes(include=[np.number]).columns
        results_df[numeric_cols] = results_df[numeric_cols].round(4)
        
        print(results_df.to_string(index=False))
        print("="*80)
        
        # Print best model
        if 'model' in results_df.columns:
            best_model = results_df.iloc[0]['model']
            print(f"\nBest performing model: {best_model}")
            
            # Print top metrics
            metric_cols = [col for col in results_df.columns if col != 'model']
            if metric_cols:
                best_metrics = results_df.iloc[0][metric_cols]
                print("\nTop metrics:")
                for metric, value in best_metrics.items():
                    if not pd.isna(value):
                        print(f"  {metric}: {value:.4f}")
    
    def generate_report(
        self,
        models: Dict[str, BaseRecommender],
        test_dataset: MusicDataset,
        train_dataset: MusicDataset,
        save_path: str = None
    ) -> str:
        """Generate a comprehensive evaluation report.
        
        Args:
            models: Dictionary of model names to model instances.
            test_dataset: Test dataset for evaluation.
            train_dataset: Training dataset.
            save_path: Optional path to save the report.
            
        Returns:
            Report as string.
        """
        results_df = self.compare_models(models, test_dataset, train_dataset)
        
        report = []
        report.append("# Music Recommendation System Evaluation Report")
        report.append("=" * 50)
        report.append("")
        
        # Dataset statistics
        report.append("## Dataset Statistics")
        report.append(f"- Training users: {len(train_dataset.user_to_idx)}")
        report.append(f"- Training items: {len(train_dataset.item_to_idx)}")
        report.append(f"- Training interactions: {len(train_dataset.interactions_df)}")
        report.append(f"- Test users: {len(test_dataset.user_to_idx)}")
        report.append(f"- Test items: {len(test_dataset.item_to_idx)}")
        report.append(f"- Test interactions: {len(test_dataset.interactions_df)}")
        report.append("")
        
        # Model comparison
        report.append("## Model Comparison")
        report.append("")
        report.append(results_df.to_string(index=False))
        report.append("")
        
        # Best model analysis
        if len(results_df) > 0:
            best_model = results_df.iloc[0]['model']
            report.append(f"## Best Model: {best_model}")
            report.append("")
            
            best_metrics = results_df.iloc[0]
            report.append("### Top Metrics:")
            for col in results_df.columns:
                if col != 'model' and not pd.isna(best_metrics[col]):
                    report.append(f"- {col}: {best_metrics[col]:.4f}")
            report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        report.append("- Consider ensemble methods for better performance")
        report.append("- Tune hyperparameters for individual models")
        report.append("- Collect more user feedback for better training data")
        report.append("- Implement real-time model updates")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            logger.info(f"Evaluation report saved to {save_path}")
        
        return report_text

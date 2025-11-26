"""Streamlit demo for music recommendation system."""

import logging
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple

from src.data import MusicDataLoader
from src.models import ContentBasedRecommender, CollaborativeFilteringRecommender, HybridRecommender
from src.evaluation import RecommenderEvaluator
from src.utils import load_config, set_seed

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Music Recommendation System",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load configuration
try:
    config = load_config("configs/config.yaml")
except Exception as e:
    st.error(f"Failed to load configuration: {e}")
    st.stop()

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'train_dataset' not in st.session_state:
    st.session_state.train_dataset = None
if 'test_dataset' not in st.session_state:
    st.session_state.test_dataset = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = None


def load_data():
    """Load and prepare the dataset."""
    with st.spinner("Loading music dataset..."):
        try:
            data_loader = MusicDataLoader(
                data_dir=config['data']['data_dir'],
                seed=config['data']['seed']
            )
            
            # Load data
            dataset = data_loader.load_data()
            
            # Create train/test split
            train_dataset, test_dataset = data_loader.create_train_test_split(
                dataset,
                test_size=config['data']['test_size'],
                time_based=config['data']['time_based_split']
            )
            
            st.session_state.dataset = dataset
            st.session_state.train_dataset = train_dataset
            st.session_state.test_dataset = test_dataset
            st.session_state.data_loaded = True
            
            st.success("Dataset loaded successfully!")
            return True
            
        except Exception as e:
            st.error(f"Failed to load dataset: {e}")
            return False


def train_models():
    """Train all recommendation models."""
    if not st.session_state.data_loaded:
        st.error("Please load the dataset first!")
        return False
    
    with st.spinner("Training recommendation models..."):
        try:
            train_dataset = st.session_state.train_dataset
            
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
            for name, model in models.items():
                with st.spinner(f"Training {name} model..."):
                    model.fit(train_dataset)
            
            st.session_state.models = models
            st.session_state.models_trained = True
            
            st.success("All models trained successfully!")
            return True
            
        except Exception as e:
            st.error(f"Failed to train models: {e}")
            return False


def evaluate_models():
    """Evaluate all trained models."""
    if not st.session_state.models_trained:
        st.error("Please train the models first!")
        return False
    
    with st.spinner("Evaluating models..."):
        try:
            evaluator = RecommenderEvaluator(k_values=config['evaluation']['k_values'])
            
            results_df = evaluator.compare_models(
                st.session_state.models,
                st.session_state.test_dataset,
                st.session_state.train_dataset
            )
            
            st.session_state.evaluation_results = results_df
            
            st.success("Model evaluation completed!")
            return True
            
        except Exception as e:
            st.error(f"Failed to evaluate models: {e}")
            return False


def main():
    """Main Streamlit application."""
    st.title("🎵 Music Recommendation System")
    st.markdown("A modern music recommendation system using content-based filtering, collaborative filtering, and hybrid approaches.")
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Overview", "Data Exploration", "Model Training", "Recommendations", "Evaluation", "Similar Items"]
    )
    
    # Load data button
    if st.sidebar.button("🔄 Load Dataset", type="primary"):
        load_data()
    
    # Train models button
    if st.sidebar.button("🚀 Train Models"):
        train_models()
    
    # Evaluate models button
    if st.sidebar.button("📊 Evaluate Models"):
        evaluate_models()
    
    # Page routing
    if page == "Overview":
        show_overview()
    elif page == "Data Exploration":
        show_data_exploration()
    elif page == "Model Training":
        show_model_training()
    elif page == "Recommendations":
        show_recommendations()
    elif page == "Evaluation":
        show_evaluation()
    elif page == "Similar Items":
        show_similar_items()


def show_overview():
    """Show system overview."""
    st.header("System Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Dataset Statistics")
        if st.session_state.data_loaded:
            dataset = st.session_state.dataset
            st.metric("Total Users", len(dataset.user_to_idx))
            st.metric("Total Items", len(dataset.item_to_idx))
            st.metric("Total Interactions", len(dataset.interactions_df))
            
            if st.session_state.train_dataset:
                train_dataset = st.session_state.train_dataset
                test_dataset = st.session_state.test_dataset
                st.metric("Training Interactions", len(train_dataset.interactions_df))
                st.metric("Test Interactions", len(test_dataset.interactions_df))
        else:
            st.info("Load the dataset to see statistics")
    
    with col2:
        st.subheader("🤖 Model Status")
        if st.session_state.models_trained:
            st.success("✅ All models trained")
            for model_name in st.session_state.models.keys():
                st.write(f"• {model_name}")
        else:
            st.info("Train models to see status")
    
    st.subheader("🎯 Features")
    features = [
        "Content-based filtering using TF-IDF and sentence transformers",
        "Collaborative filtering with matrix factorization (ALS/BPR)",
        "Hybrid approach combining both methods",
        "Comprehensive evaluation metrics (Precision@K, Recall@K, NDCG@K)",
        "Interactive demo with real-time recommendations",
        "Similar item discovery",
        "Model comparison and performance analysis"
    ]
    
    for feature in features:
        st.write(f"• {feature}")


def show_data_exploration():
    """Show data exploration visualizations."""
    st.header("Data Exploration")
    
    if not st.session_state.data_loaded:
        st.warning("Please load the dataset first!")
        return
    
    dataset = st.session_state.dataset
    
    # Genre distribution
    st.subheader("📈 Genre Distribution")
    genre_counts = dataset.items_df['genre'].value_counts()
    fig_genre = px.pie(
        values=genre_counts.values,
        names=genre_counts.index,
        title="Distribution of Music Genres"
    )
    st.plotly_chart(fig_genre, use_container_width=True)
    
    # Interaction patterns
    st.subheader("📊 Interaction Patterns")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # User interaction counts
        user_interactions = dataset.interactions_df.groupby('user_id').size()
        fig_user = px.histogram(
            x=user_interactions.values,
            title="Distribution of User Interactions",
            labels={'x': 'Number of Interactions', 'y': 'Number of Users'}
        )
        st.plotly_chart(fig_user, use_container_width=True)
    
    with col2:
        # Item popularity
        item_interactions = dataset.interactions_df.groupby('item_id')['weight'].sum().sort_values(ascending=False)
        top_items = item_interactions.head(20)
        
        fig_items = px.bar(
            x=top_items.values,
            y=top_items.index,
            orientation='h',
            title="Top 20 Most Popular Items",
            labels={'x': 'Total Weight', 'y': 'Item ID'}
        )
        st.plotly_chart(fig_items, use_container_width=True)
    
    # Temporal patterns
    st.subheader("⏰ Temporal Patterns")
    dataset.interactions_df['date'] = pd.to_datetime(dataset.interactions_df['timestamp']).dt.date
    daily_interactions = dataset.interactions_df.groupby('date').size()
    
    fig_time = px.line(
        x=daily_interactions.index,
        y=daily_interactions.values,
        title="Daily Interaction Count",
        labels={'x': 'Date', 'y': 'Number of Interactions'}
    )
    st.plotly_chart(fig_time, use_container_width=True)
    
    # Raw data preview
    st.subheader("📋 Data Preview")
    
    tab1, tab2, tab3 = st.tabs(["Interactions", "Items", "Users"])
    
    with tab1:
        st.dataframe(dataset.interactions_df.head(10))
    
    with tab2:
        st.dataframe(dataset.items_df.head(10))
    
    with tab3:
        if dataset.users_df is not None:
            st.dataframe(dataset.users_df.head(10))
        else:
            st.info("No user data available")


def show_model_training():
    """Show model training interface."""
    st.header("Model Training")
    
    if not st.session_state.data_loaded:
        st.warning("Please load the dataset first!")
        return
    
    st.subheader("Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Content-Based Model**")
        st.write(f"• Use Sentence Transformer: {config['models']['content_based']['use_sentence_transformer']}")
        st.write(f"• Seed: {config['models']['content_based']['seed']}")
        
        st.write("**Collaborative Filtering Model**")
        st.write(f"• Model Type: {config['models']['collaborative_filtering']['model_type']}")
        st.write(f"• Factors: {config['models']['collaborative_filtering']['factors']}")
        st.write(f"• Regularization: {config['models']['collaborative_filtering']['regularization']}")
        st.write(f"• Iterations: {config['models']['collaborative_filtering']['iterations']}")
    
    with col2:
        st.write("**Hybrid Model**")
        st.write(f"• Content Weight: {config['models']['hybrid']['content_weight']}")
        st.write(f"• Collaborative Weight: {config['models']['hybrid']['collab_weight']}")
        
        st.write("**Training Data**")
        train_dataset = st.session_state.train_dataset
        st.write(f"• Training Users: {len(train_dataset.user_to_idx)}")
        st.write(f"• Training Items: {len(train_dataset.item_to_idx)}")
        st.write(f"• Training Interactions: {len(train_dataset.interactions_df)}")
    
    if st.session_state.models_trained:
        st.success("✅ All models have been trained successfully!")
        
        # Show training progress (simulated)
        st.subheader("Training Progress")
        
        models = ['Content-Based', 'Collaborative Filtering', 'Hybrid']
        progress = [1.0, 1.0, 1.0]  # All completed
        
        for model, prog in zip(models, progress):
            st.progress(prog, text=f"{model}: {prog*100:.0f}%")
    else:
        st.info("Click 'Train Models' in the sidebar to start training")


def show_recommendations():
    """Show recommendation interface."""
    st.header("Music Recommendations")
    
    if not st.session_state.models_trained:
        st.warning("Please train the models first!")
        return
    
    dataset = st.session_state.dataset
    models = st.session_state.models
    
    # User selection
    st.subheader("Select User")
    user_options = list(dataset.user_to_idx.keys())
    selected_user = st.selectbox("Choose a user:", user_options)
    
    if selected_user:
        # Show user's interaction history
        user_interactions = dataset.get_user_interactions(selected_user)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**User's Music History**")
            if len(user_interactions) > 0:
                # Get item details for user's interactions
                user_items = user_interactions.merge(
                    dataset.items_df, 
                    on='item_id', 
                    how='left'
                )[['title', 'artist', 'genre', 'weight']].head(10)
                st.dataframe(user_items)
            else:
                st.info("No interaction history for this user")
        
        with col2:
            st.write("**User Profile**")
            if dataset.users_df is not None:
                user_profile = dataset.get_user_features(selected_user)
                if user_profile is not None:
                    st.write(f"• Age: {user_profile['age']}")
                    st.write(f"• Gender: {user_profile['gender']}")
                    st.write(f"• Location: {user_profile['location']}")
                    st.write(f"• Preferred Genres: {user_profile['preferred_genres']}")
                    st.write(f"• Subscription: {user_profile['subscription_type']}")
            else:
                st.info("No user profile data available")
    
    # Model selection and recommendations
    st.subheader("Get Recommendations")
    
    selected_model = st.selectbox("Choose a model:", list(models.keys()))
    n_recommendations = st.slider("Number of recommendations:", 5, 20, config['demo']['n_recommendations'])
    
    if st.button("Get Recommendations", type="primary"):
        if selected_user and selected_model:
            try:
                model = models[selected_model]
                recommendations = model.recommend(
                    selected_user, 
                    n_recommendations=n_recommendations,
                    exclude_seen=True
                )
                
                if recommendations:
                    st.success(f"Top {len(recommendations)} recommendations from {selected_model}:")
                    
                    # Create recommendations dataframe
                    rec_data = []
                    for i, (item_id, score) in enumerate(recommendations, 1):
                        item_info = dataset.get_item_features(item_id)
                        rec_data.append({
                            'Rank': i,
                            'Title': item_info['title'],
                            'Artist': item_info['artist'],
                            'Genre': item_info['genre'],
                            'Score': f"{score:.4f}"
                        })
                    
                    rec_df = pd.DataFrame(rec_data)
                    st.dataframe(rec_df, use_container_width=True)
                    
                    # Show recommendation explanation
                    st.subheader("Why These Recommendations?")
                    if selected_model == "Content-Based":
                        st.write("These recommendations are based on the similarity between your music preferences and the content features of songs (genre, mood, tempo, etc.).")
                    elif selected_model == "Collaborative Filtering":
                        st.write("These recommendations are based on users with similar music tastes to you and what they have enjoyed.")
                    elif selected_model == "Hybrid":
                        st.write("These recommendations combine both content-based and collaborative filtering approaches for better accuracy.")
                else:
                    st.warning("No recommendations available for this user")
                    
            except Exception as e:
                st.error(f"Failed to get recommendations: {e}")


def show_evaluation():
    """Show model evaluation results."""
    st.header("Model Evaluation")
    
    if not st.session_state.models_trained:
        st.warning("Please train the models first!")
        return
    
    if st.session_state.evaluation_results is None:
        st.info("Click 'Evaluate Models' in the sidebar to see evaluation results")
        return
    
    results_df = st.session_state.evaluation_results
    
    st.subheader("📊 Performance Metrics")
    
    # Display results table
    st.dataframe(results_df, use_container_width=True)
    
    # Create visualizations
    st.subheader("📈 Performance Comparison")
    
    # Select metrics to visualize
    metric_cols = [col for col in results_df.columns if col != 'model']
    selected_metrics = st.multiselect(
        "Select metrics to visualize:",
        metric_cols,
        default=metric_cols[:4] if len(metric_cols) >= 4 else metric_cols
    )
    
    if selected_metrics:
        # Create bar chart
        fig = go.Figure()
        
        for metric in selected_metrics:
            fig.add_trace(go.Bar(
                name=metric,
                x=results_df['model'],
                y=results_df[metric],
                text=results_df[metric].round(4),
                textposition='auto'
            ))
        
        fig.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Model",
            yaxis_title="Score",
            barmode='group'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Create radar chart for top model
        if len(results_df) > 0:
            best_model = results_df.iloc[0]
            
            # Select metrics for radar chart (exclude coverage metrics)
            radar_metrics = [col for col in selected_metrics if 'coverage' not in col.lower()]
            
            if len(radar_metrics) >= 3:
                fig_radar = go.Figure()
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=[best_model[metric] for metric in radar_metrics],
                    theta=radar_metrics,
                    fill='toself',
                    name=best_model['model']
                ))
                
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )),
                    title=f"Performance Radar Chart - {best_model['model']}"
                )
                
                st.plotly_chart(fig_radar, use_container_width=True)
    
    # Model insights
    st.subheader("🔍 Model Insights")
    
    if len(results_df) > 0:
        best_model = results_df.iloc[0]
        st.success(f"**Best Model:** {best_model['model']}")
        
        # Show top metrics
        top_metrics = []
        for col in results_df.columns:
            if col != 'model' and not pd.isna(best_model[col]):
                top_metrics.append(f"• {col}: {best_model[col]:.4f}")
        
        if top_metrics:
            st.write("**Top Performance Metrics:**")
            for metric in top_metrics[:5]:  # Show top 5 metrics
                st.write(metric)


def show_similar_items():
    """Show similar items interface."""
    st.header("Similar Items Discovery")
    
    if not st.session_state.models_trained:
        st.warning("Please train the models first!")
        return
    
    dataset = st.session_state.dataset
    models = st.session_state.models
    
    # Item selection
    st.subheader("Select Item")
    item_options = list(dataset.item_to_idx.keys())
    selected_item = st.selectbox("Choose an item:", item_options)
    
    if selected_item:
        # Show item details
        item_info = dataset.get_item_features(selected_item)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Item Details**")
            st.write(f"• Title: {item_info['title']}")
            st.write(f"• Artist: {item_info['artist']}")
            st.write(f"• Genre: {item_info['genre']}")
            st.write(f"• Mood: {item_info['mood']}")
            st.write(f"• Tempo: {item_info['tempo']} BPM")
            st.write(f"• Year: {item_info['year']}")
        
        with col2:
            st.write("**Description**")
            st.write(item_info['description'])
    
    # Model selection and similar items
    st.subheader("Find Similar Items")
    
    # Only show models that support similar items
    available_models = [name for name, model in models.items() 
                       if hasattr(model, 'get_similar_items')]
    
    if not available_models:
        st.warning("No models available that support similar item discovery")
        return
    
    selected_model = st.selectbox("Choose a model:", available_models)
    n_similar = st.slider("Number of similar items:", 5, 20, config['demo']['n_similar_items'])
    
    if st.button("Find Similar Items", type="primary"):
        if selected_item and selected_model:
            try:
                model = models[selected_model]
                similar_items = model.get_similar_items(selected_item, n_similar=n_similar)
                
                if similar_items:
                    st.success(f"Top {len(similar_items)} similar items:")
                    
                    # Create similar items dataframe
                    similar_data = []
                    for i, (item_id, score) in enumerate(similar_items, 1):
                        item_info = dataset.get_item_features(item_id)
                        similar_data.append({
                            'Rank': i,
                            'Title': item_info['title'],
                            'Artist': item_info['artist'],
                            'Genre': item_info['genre'],
                            'Similarity': f"{score:.4f}"
                        })
                    
                    similar_df = pd.DataFrame(similar_data)
                    st.dataframe(similar_df, use_container_width=True)
                    
                    # Show similarity explanation
                    st.subheader("Why These Items Are Similar?")
                    if selected_model == "Content-Based":
                        st.write("These items are similar based on their content features like genre, mood, tempo, and description.")
                    elif selected_model == "Hybrid":
                        st.write("These items are similar based on content features and user interaction patterns.")
                else:
                    st.warning("No similar items found")
                    
            except Exception as e:
                st.error(f"Failed to find similar items: {e}")


if __name__ == "__main__":
    main()

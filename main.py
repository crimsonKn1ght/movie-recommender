import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.metrics import mean_squared_error, mean_absolute_error
import requests
import zipfile
import os
import warnings
warnings.filterwarnings('ignore')

# For neural collaborative filtering
try:
    import tensorflow
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.regularizers import l2
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. Neural CF will be disabled.")

class AdvancedMovieRecommender:
    def __init__(self):
        self.ratings = None
        self.movies = None
        self.user_item_matrix = None
        self.movie_similarity = None
        
        # Advanced models
        self.svd_model = None
        self.nmf_model = None
        self.neural_model = None
        
        # Evaluation data
        self.train_data = None
        self.test_data = None
        
        # Mappings for neural model
        self.user_to_idx = {}
        self.movie_to_idx = {}
        self.idx_to_user = {}
        self.idx_to_movie = {}
        
    def download_movielens_data(self):
        """Download and extract MovieLens 100K dataset"""
        url = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
        
        print("Downloading MovieLens 100K dataset...")
        response = requests.get(url)
        
        with open("ml-100k.zip", "wb") as f:
            f.write(response.content)
        
        with zipfile.ZipFile("ml-100k.zip", 'r') as zip_ref:
            zip_ref.extractall(".")
        
        print("Dataset downloaded and extracted!")
        
    def load_data(self):
        """Load the MovieLens dataset"""
        if not os.path.exists("ml-100k"):
            self.download_movielens_data()
        
        self.ratings = pd.read_csv(
            'ml-100k/u.data', 
            sep='\t', 
            names=['user_id', 'movie_id', 'rating', 'timestamp']
        )
        
        self.movies = pd.read_csv(
            'ml-100k/u.item', 
            sep='|', 
            encoding='latin-1',
            names=['movie_id', 'title', 'release_date', 'video_release_date', 
                   'imdb_url', 'unknown', 'Action', 'Adventure', 'Animation',
                   'Children', 'Comedy', 'Crime', 'Documentary', 'Drama',
                   'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
                   'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
        )
        
        print(f"Loaded {len(self.ratings)} ratings and {len(self.movies)} movies")
        return self.ratings, self.movies
        
    def create_train_test_split(self, test_size=0.2):
        """Split data for evaluation"""
        if self.ratings is None:
            self.load_data()
        
        self.train_data, self.test_data = train_test_split(
            self.ratings, test_size=test_size, random_state=42
        )
        
        print(f"Train set: {len(self.train_data)} ratings")
        print(f"Test set: {len(self.test_data)} ratings")
        
        # Create user-item matrix from training data only
        self.user_item_matrix = self.train_data.pivot_table(
            index='user_id', 
            columns='movie_id', 
            values='rating'
        ).fillna(0)
        
        return self.train_data, self.test_data
    
    def build_svd_model(self, n_components=50):
        """
        Matrix Factorization using SVD (Singular Value Decomposition)
        Decomposes the user-item matrix into lower dimensional representations
        """
        if self.user_item_matrix is None:
            self.create_train_test_split()
        
        print(f"Building SVD model with {n_components} components...")
        
        # SVD works better with mean-centered data
        self.user_means = self.user_item_matrix.mean(axis=1)
        user_item_centered = self.user_item_matrix.sub(self.user_means, axis=0).fillna(0)
        
        self.svd_model = TruncatedSVD(n_components=n_components, random_state=42)
        self.user_factors = self.svd_model.fit_transform(user_item_centered)
        self.movie_factors = self.svd_model.components_.T
        
        print(f"SVD model built! User factors shape: {self.user_factors.shape}")
        print(f"Movie factors shape: {self.movie_factors.shape}")
        
        return self.svd_model
    
    def build_nmf_model(self, n_components=50):
        """
        Matrix Factorization using NMF (Non-negative Matrix Factorization)
        Better for sparse data and provides interpretable factors
        """
        if self.user_item_matrix is None:
            self.create_train_test_split()
        
        print(f"Building NMF model with {n_components} components...")
        
        # NMF requires non-negative data
        user_item_positive = self.user_item_matrix.copy()
        
        self.nmf_model = NMF(n_components=n_components, random_state=42, max_iter=1000)
        self.user_factors_nmf = self.nmf_model.fit_transform(user_item_positive)
        self.movie_factors_nmf = self.nmf_model.components_.T
        
        print(f"NMF model built! User factors shape: {self.user_factors_nmf.shape}")
        print(f"Movie factors shape: {self.movie_factors_nmf.shape}")
        
        return self.nmf_model
    
    def build_neural_cf_model(self, embedding_size=50, hidden_units=[64, 32]):
        """
        Neural Collaborative Filtering
        Uses deep learning to learn complex user-item interactions
        """
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow not available. Cannot build Neural CF model.")
            return None
            
        if self.train_data is None:
            self.create_train_test_split()
        
        print("Building Neural Collaborative Filtering model...")
        
        # Create mappings
        unique_users = self.train_data['user_id'].unique()
        unique_movies = self.train_data['movie_id'].unique()
        
        self.user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        self.movie_to_idx = {movie: idx for idx, movie in enumerate(unique_movies)}
        self.idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}
        self.idx_to_movie = {idx: movie for movie, idx in self.movie_to_idx.items()}
        
        n_users = len(unique_users)
        n_movies = len(unique_movies)
        
        # Build neural network architecture
        user_input = Input(shape=(), name='user_id')
        movie_input = Input(shape=(), name='movie_id')
        
        # Embedding layers
        user_embedding = Embedding(n_users, embedding_size, 
                                 embeddings_regularizer=l2(1e-6))(user_input)
        movie_embedding = Embedding(n_movies, embedding_size,
                                  embeddings_regularizer=l2(1e-6))(movie_input)
        
        user_vec = Flatten()(user_embedding)
        movie_vec = Flatten()(movie_embedding)
        
        # Concatenate user and movie vectors
        concat = Concatenate()([user_vec, movie_vec])
        
        # Hidden layers
        x = concat
        for units in hidden_units:
            x = Dense(units, activation='relu', 
                     kernel_regularizer=l2(1e-6))(x)
            x = Dropout(0.2)(x)
        
        # Output layer
        output = Dense(1, activation='linear')(x)
        
        # Create and compile model
        self.neural_model = Model(inputs=[user_input, movie_input], outputs=output)
        self.neural_model.compile(optimizer=Adam(learning_rate=0.001),
                                loss='mse', metrics=['mae'])
        
        print(f"Neural CF model created with {n_users} users and {n_movies} movies")
        return self.neural_model
    
    def train_neural_model(self, epochs=50, batch_size=256):
        """Train the neural collaborative filtering model"""
        if self.neural_model is None:
            self.build_neural_cf_model()
        
        if self.neural_model is None:
            return None
        
        # Prepare training data
        train_users = [self.user_to_idx[user] for user in self.train_data['user_id']]
        train_movies = [self.movie_to_idx[movie] for movie in self.train_data['movie_id']]
        train_ratings = self.train_data['rating'].values
        
        print("Training Neural CF model...")
        history = self.neural_model.fit(
            [np.array(train_users), np.array(train_movies)],
            train_ratings,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            validation_split=0.1
        )
        
        return history
    
    def get_svd_recommendations(self, user_id, n_recommendations=10):
        """Get recommendations using SVD matrix factorization"""
        if self.svd_model is None:
            self.build_svd_model()
        
        if user_id not in self.user_item_matrix.index:
            print(f"User {user_id} not found!")
            return None
        
        user_idx = list(self.user_item_matrix.index).index(user_id)
        user_mean = self.user_means.iloc[user_idx]
        
        # Predict ratings for all movies
        predicted_ratings = np.dot(self.user_factors[user_idx], self.movie_factors.T) + user_mean
        
        # Get movies user hasn't rated
        user_ratings = self.user_item_matrix.iloc[user_idx]
        unrated_movies = user_ratings[user_ratings == 0].index
        
        # Get predictions for unrated movies
        movie_predictions = []
        for movie_id in unrated_movies:
            if movie_id in self.user_item_matrix.columns:
                movie_idx = list(self.user_item_matrix.columns).index(movie_id)
                predicted_rating = predicted_ratings[movie_idx]
                movie_predictions.append((movie_id, predicted_rating))
        
        # Sort by predicted rating
        movie_predictions.sort(key=lambda x: x[1], reverse=True)
        top_predictions = movie_predictions[:n_recommendations]
        
        # Format recommendations
        recommendations = []
        for movie_id, predicted_rating in top_predictions:
            movie_info = self.movies[self.movies['movie_id'] == movie_id].iloc[0]
            recommendations.append({
                'title': movie_info['title'],
                'predicted_rating': predicted_rating,
                'movie_id': movie_id,
                'method': 'SVD'
            })
        
        return recommendations
    
    def get_nmf_recommendations(self, user_id, n_recommendations=10):
        """Get recommendations using NMF matrix factorization"""
        if self.nmf_model is None:
            self.build_nmf_model()
        
        if user_id not in self.user_item_matrix.index:
            print(f"User {user_id} not found!")
            return None
        
        user_idx = list(self.user_item_matrix.index).index(user_id)
        
        # Predict ratings for all movies
        predicted_ratings = np.dot(self.user_factors_nmf[user_idx], self.movie_factors_nmf.T)
        
        # Get movies user hasn't rated
        user_ratings = self.user_item_matrix.iloc[user_idx]
        unrated_movies = user_ratings[user_ratings == 0].index
        
        # Get predictions for unrated movies
        movie_predictions = []
        for movie_id in unrated_movies:
            if movie_id in self.user_item_matrix.columns:
                movie_idx = list(self.user_item_matrix.columns).index(movie_id)
                predicted_rating = predicted_ratings[movie_idx]
                movie_predictions.append((movie_id, predicted_rating))
        
        # Sort by predicted rating
        movie_predictions.sort(key=lambda x: x[1], reverse=True)
        top_predictions = movie_predictions[:n_recommendations]
        
        # Format recommendations
        recommendations = []
        for movie_id, predicted_rating in top_predictions:
            movie_info = self.movies[self.movies['movie_id'] == movie_id].iloc[0]
            recommendations.append({
                'title': movie_info['title'],
                'predicted_rating': predicted_rating,
                'movie_id': movie_id,
                'method': 'NMF'
            })
        
        return recommendations
    
    def get_neural_recommendations(self, user_id, n_recommendations=10):
        """Get recommendations using Neural Collaborative Filtering"""
        if self.neural_model is None or user_id not in self.user_to_idx:
            print("Neural model not available or user not found!")
            return None
        
        user_idx = self.user_to_idx[user_id]
        
        # Get all movies user hasn't rated
        user_rated_movies = set(self.train_data[self.train_data['user_id'] == user_id]['movie_id'])
        all_movies = set(self.movie_to_idx.keys())
        unrated_movies = all_movies - user_rated_movies
        
        if not unrated_movies:
            return []
        
        # Prepare data for prediction
        user_array = np.array([user_idx] * len(unrated_movies))
        movie_array = np.array([self.movie_to_idx[movie] for movie in unrated_movies])
        
        # Get predictions
        predictions = self.neural_model.predict([user_array, movie_array], verbose=0)
        
        # Combine movies with predictions
        movie_predictions = list(zip(unrated_movies, predictions.flatten()))
        movie_predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Format recommendations
        recommendations = []
        for movie_id, predicted_rating in movie_predictions[:n_recommendations]:
            movie_info = self.movies[self.movies['movie_id'] == movie_id].iloc[0]
            recommendations.append({
                'title': movie_info['title'],
                'predicted_rating': predicted_rating,
                'movie_id': movie_id,
                'method': 'Neural CF'
            })
        
        return recommendations
    
    def evaluate_models(self):
        """Evaluate all models on test data"""
        if self.test_data is None:
            print("No test data available. Run create_train_test_split() first.")
            return None
        
        print("Evaluating models on test data...")
        results = {}
        
        # Evaluate SVD
        if self.svd_model is not None:
            svd_predictions = []
            actual_ratings = []
            
            for _, row in self.test_data.iterrows():
                user_id, movie_id, actual_rating = row['user_id'], row['movie_id'], row['rating']
                
                if (user_id in self.user_item_matrix.index and 
                    movie_id in self.user_item_matrix.columns):
                    
                    user_idx = list(self.user_item_matrix.index).index(user_id)
                    movie_idx = list(self.user_item_matrix.columns).index(movie_id)
                    user_mean = self.user_means.iloc[user_idx]
                    
                    predicted = np.dot(self.user_factors[user_idx], 
                                     self.movie_factors[movie_idx]) + user_mean
                    
                    svd_predictions.append(predicted)
                    actual_ratings.append(actual_rating)
            
            if svd_predictions:
                svd_rmse = np.sqrt(mean_squared_error(actual_ratings, svd_predictions))
                svd_mae = mean_absolute_error(actual_ratings, svd_predictions)
                results['SVD'] = {'RMSE': svd_rmse, 'MAE': svd_mae}
        
        # Evaluate Neural CF
        if self.neural_model is not None:
            neural_predictions = []
            actual_ratings = []
            
            for _, row in self.test_data.iterrows():
                user_id, movie_id, actual_rating = row['user_id'], row['movie_id'], row['rating']
                
                if user_id in self.user_to_idx and movie_id in self.movie_to_idx:
                    user_idx = self.user_to_idx[user_id]
                    movie_idx = self.movie_to_idx[movie_id]
                    
                    predicted = self.neural_model.predict(
                        [np.array([user_idx]), np.array([movie_idx])], verbose=0
                    )[0][0]
                    
                    neural_predictions.append(predicted)
                    actual_ratings.append(actual_rating)
            
            if neural_predictions:
                neural_rmse = np.sqrt(mean_squared_error(actual_ratings, neural_predictions))
                neural_mae = mean_absolute_error(actual_ratings, neural_predictions)
                results['Neural CF'] = {'RMSE': neural_rmse, 'MAE': neural_mae}
        
        # Print results
        for model_name, metrics in results.items():
            print(f"\n{model_name} Performance:")
            print(f"  RMSE: {metrics['RMSE']:.4f}")
            print(f"  MAE: {metrics['MAE']:.4f}")
        
        return results
    
    def compare_recommendations(self, user_id, n_recommendations=5):
        """Compare recommendations from all available models"""
        print(f"\n=== Recommendations for User {user_id} ===")
        
        # SVD recommendations
        svd_recs = self.get_svd_recommendations(user_id, n_recommendations)
        if svd_recs:
            print(f"\nSVD Matrix Factorization:")
            for i, rec in enumerate(svd_recs, 1):
                print(f"  {i}. {rec['title']} (rating: {rec['predicted_rating']:.2f})")
        
        # NMF recommendations
        nmf_recs = self.get_nmf_recommendations(user_id, n_recommendations)
        if nmf_recs:
            print(f"\nNMF Matrix Factorization:")
            for i, rec in enumerate(nmf_recs, 1):
                print(f"  {i}. {rec['title']} (rating: {rec['predicted_rating']:.2f})")
        
        # Neural CF recommendations
        neural_recs = self.get_neural_recommendations(user_id, n_recommendations)
        if neural_recs:
            print(f"\nNeural Collaborative Filtering:")
            for i, rec in enumerate(neural_recs, 1):
                print(f"  {i}. {rec['title']} (rating: {rec['predicted_rating']:.2f})")


# Example usage
if __name__ == "__main__":
    # Initialize advanced recommender
    recommender = AdvancedMovieRecommender()
    
    # Load data and create train/test split
    print("Loading MovieLens 100K dataset...")
    recommender.load_data()
    recommender.create_train_test_split()
    
    # Build all models
    print("\n=== Building Advanced Models ===")
    
    # 1. SVD Matrix Factorization
    recommender.build_svd_model(n_components=50)
    
    # 2. NMF Matrix Factorization
    recommender.build_nmf_model(n_components=50)
    
    # 3. Neural Collaborative Filtering (if TensorFlow available)
    if TENSORFLOW_AVAILABLE:
        recommender.build_neural_cf_model(embedding_size=50)
        recommender.train_neural_model(epochs=20, batch_size=256)
    
    # Compare recommendations from all models
    recommender.compare_recommendations(user_id=1, n_recommendations=5)
    
    # Evaluate model performance
    print("\n=== Model Evaluation ===")
    recommender.evaluate_models()
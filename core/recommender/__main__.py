"""Command-line demo for the recommendation engine.

Run with ``python -m recommender`` from the ``core`` directory (or after
``pip install -e core``). Mirrors the original ``main.py`` example usage.
"""

from .engine import TENSORFLOW_AVAILABLE, AdvancedMovieRecommender


def main():
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


if __name__ == "__main__":
    main()

"""Movie recommendation core engine.

Pure-Python recommendation logic, independent of any web framework. The backend
imports from here; the package can also be exercised directly via
``python -m recommender``.
"""

from .engine import TENSORFLOW_AVAILABLE, AdvancedMovieRecommender

__all__ = ["AdvancedMovieRecommender", "TENSORFLOW_AVAILABLE"]

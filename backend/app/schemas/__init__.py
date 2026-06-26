"""Pydantic request/response models shared across routers."""

from pydantic import BaseModel


class Metrics(BaseModel):
    rmse: float
    mae: float


class Algorithm(BaseModel):
    key: str
    name: str
    description: str
    status: str
    metrics: Metrics | None = None


class Movie(BaseModel):
    movie_id: int
    title: str
    year: int | None = None
    genres: list[str] = []
    avg_rating: float | None = None


class Recommendation(Movie):
    predicted_rating: float


class TrainRequest(BaseModel):
    algorithm: str = "all"  # "SVD" | "NMF" | "Neural" | "all"

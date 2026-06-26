"""Non-secret runtime configuration, read from environment variables.

Mirrors the documentation's split between server config and code: values live in
the environment (or a local ``.env``) and are read once at import time.
"""

import os


def _bool(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in {"1", "true", "yes"}


class Settings:
    # Comma-separated list of browser origins allowed by CORS.
    ALLOWED_ORIGINS = [
        o.strip()
        for o in os.environ.get("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
        if o.strip()
    ]

    # Train the fast (SVD + NMF) models in a background thread at startup so the
    # first /recommend call does not pay the training cost.
    AUTO_TRAIN_ON_STARTUP = _bool("AUTO_TRAIN_ON_STARTUP", True)

    # Model hyperparameters.
    SVD_COMPONENTS = int(os.environ.get("SVD_COMPONENTS", "50"))
    NMF_COMPONENTS = int(os.environ.get("NMF_COMPONENTS", "50"))
    NEURAL_EMBEDDING_SIZE = int(os.environ.get("NEURAL_EMBEDDING_SIZE", "50"))
    NEURAL_EPOCHS = int(os.environ.get("NEURAL_EPOCHS", "20"))


settings = Settings()

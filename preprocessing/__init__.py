"""
Description: 
--------------
    Preprocessing Package Initialization
    ====================================

    This package handles dataset preprocessing for the IMDb sentiment analysis project.
    It includes:
    - Data loading (`load_imdb_data`)
    - Padding sequences (`pad_reviews`)
    - Dataset statistics logging (`log_dataset_statistics`)

Modules:
--------
- load_data.py
- pad_sequences.py
- dataset_stats.py

Authors:
--------
- Ehiane Oigiagbe
- Osaze Ogieriakhi

Created: March 19, 2025
"""
from .load_data import load_imdb_data;
from .pad_sequences import pad_reviews;
from .dataset_stats import log_dataset_statistics;
from .embedding import load_glove_embeddings, create_embedding_matrix; 

# this defines the public interface of this module (controlls what can be imported from this module)
__all__ = ["load_imdb_data", "pad_reviews", "log_dataset_statistics", "load_glove_embeddings", "create_embedding_matrix"];
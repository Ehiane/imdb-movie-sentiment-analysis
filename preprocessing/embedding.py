"""
Description:
------------
    GloVe Word Embeddings for Sentiment Analysis
    ============================================
    This module loads GloVe embeddings and creates an embedding matrix
    for IMDb sentiment analysis. It also allows for experimenting with 
    trainable embeddings.

Functions:
----------
- load_glove_embeddings(glove_path): Loads GloVe word embeddings into a dictionary.
- create_embedding_matrix(vocab_size=10000, embedding_dim=100, glove_path="data/glove.6B.100d.txt"):
    Constructs an embedding matrix that aligns IMDb words with GloVe vectors.
- create_trainable_embedding(vocab_size=10000, embedding_dim=100):
    Creates a randomly initialized trainable embedding matrix.

Authors:
--------
- Ehiane Oigiagbe
- Osaze Ogieriakhi

Created: March 19, 2025
"""

import numpy as np
import tensorflow as tf

# Load IMDb dataset word index
imdb = tf.keras.datasets.imdb

def load_glove_embeddings(glove_path):
    """
    Loads pretrained GloVe word embeddings from a given file.

    Parameters:
        glove_path (str): Path to the GloVe embeddings file.

    Returns:
        dict: A dictionary mapping words to their corresponding GloVe vectors.
    """
    print("🔄 Loading GloVe embeddings...")
    
    embeddings_index = {}

    with open(glove_path, encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]  # First word in the line
            vector = np.asarray(values[1:], dtype="float32")  # Remaining are the vector values
            embeddings_index[word] = vector

    print(f"✅ Loaded {len(embeddings_index):,} word vectors from GloVe.")
    return embeddings_index


def create_embedding_matrix(vocab_size=10000, embedding_dim=100, glove_path="data/glove.6B.100d.txt", normalize=True):
    """
    Creates an embedding matrix where each row corresponds to a word in IMDb's vocabulary
    and is initialized with the corresponding GloVe vector.

    Parameters:
        vocab_size (int): Number of words to include from IMDb dataset.
        embedding_dim (int): Dimensionality of the word vectors.
        glove_path (str): Path to the GloVe embeddings file.
        normalize (bool): Whether to normalize embeddings (default=True).

    Returns:
        numpy.ndarray: The embedding matrix to be used in the model.
    """
    print("🔄 Creating embedding matrix...")

    # Load IMDb word index
    word_index = imdb.get_word_index()

    # Load GloVe embeddings
    embeddings_index = load_glove_embeddings(glove_path)

    # Initialize an empty embedding matrix
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    # Populate embedding matrix with GloVe vectors
    for word, index in word_index.items():
        if index < vocab_size:
            embedding_vector = embeddings_index.get(word)  # Retrieve GloVe vector
            if embedding_vector is not None:
                if normalize:
                    embedding_vector /= np.linalg.norm(embedding_vector)  # Normalize the vector
                embedding_matrix[index] = embedding_vector  # Assign to embedding matrix

    print(f"✅ Created embedding matrix with shape {embedding_matrix.shape}.")
    return embedding_matrix


def create_trainable_embedding(vocab_size=10000, embedding_dim=100):
    """
    Creates a randomly initialized trainable embedding matrix.

    Parameters:
        vocab_size (int): Number of words to include from IMDb dataset.
        embedding_dim (int): Dimensionality of the word vectors.

    Returns:
        numpy.ndarray: Randomly initialized trainable embedding matrix.
    """
    print("🔄 Creating trainable embedding matrix...")
    embedding_matrix = np.random.uniform(-0.05, 0.05, (vocab_size, embedding_dim))  # Small random values
    print(f"✅ Created trainable embedding matrix with shape {embedding_matrix.shape}.")
    return embedding_matrix
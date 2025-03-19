"""
Description:
------------
    Test IMDb Dataset Loading
    =========================

    This script tests whether the `load_imdb_data` function correctly loads 
    the IMDb dataset and returns expected outputs.

Author:
--------
- Ehiane Oigiagbe

Created: March 19, 2025
"""

import tensorflow as tf


def load_imdb_data(num_words=10_000):
    """
    Description:
    -----------
    Loads the IMDb dataset, containing 50,000 movie reviews labeled as 
    positive (1) or negative (0). Reviews are preprocessed into sequences 
    of numbers representing word indices.

    Parameters:
    -----------
    num_words : int, optional (default=10000)
        The maximum number of unique words to keep, based on word frequency.

    Returns:
    --------
    tuple: (training_set_reviews, training_set_labels, test_set_reviews, y_test)
        - training_set_reviews (list of lists): Tokenized training reviews.
        - training_set_labels (list): Labels for training reviews (0 = negative, 1 = positive).
        - test_set_reviews (list of lists): Tokenized test reviews.
        - test_set_labels (list): Labels for test reviews.
    """
    imdb = tf.keras.datasets.imdb;
    (training_set_reviews, training_set_labels), (test_set_reviews, test_set_labels) = imdb.load_data(num_words=num_words);
    return training_set_reviews, training_set_labels, test_set_reviews, test_set_labels;


import numpy as np

def test_Load_imdb_data():
    """
    Testing out the function (It worked)
    ----------------------------------------
    """

    # Load dataset using the function
    X_train, y_train, X_test, y_test = load_imdb_data()

    # Print basic dataset info
    print(f"Training Samples: {len(X_train)}")
    print(f"Testing Samples: {len(X_test)}\n")

    # Check label distribution with clear formatting
    train_labels, train_counts = np.unique(y_train, return_counts=True)
    test_labels, test_counts = np.unique(y_test, return_counts=True)

    print("Training Set Label Distribution:")
    print(f"  Negative Reviews (0): {train_counts[0]}")
    print(f"  Positive Reviews (1): {train_counts[1]}\n")

    print("Testing Set Label Distribution:")
    print(f"  Negative Reviews (0): {test_counts[0]}")
    print(f"  Positive Reviews (1): {test_counts[1]}\n")

    # Print a sample review in tokenized form (first 10 words)
    print("Sample Review (First 10 Words):", X_train[0][:10])
    print(f"Corresponding Sentiment Label: {y_train[0]} (0 = Negative, 1 = Positive)")

# test_Load_imdb_data()
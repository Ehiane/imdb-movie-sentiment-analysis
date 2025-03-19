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
    Dynamically accesses and loads the IMDb dataset with the top 'num_words' most frequent words.

    Returns:
        X_train, y_train, X_test, y_test: Tokenized reviews and their labels.
    """
    imdb = tf.keras.datasets.imdb;
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=num_words);
    return X_train, y_train, X_test, y_test;


def test_Load_imdb_data():
    """
    Testing out the function (It worked)
    ----------------------------------------
    """

    # Load dataset using the function
    X_train, y_train, X_test, y_test = load_imdb_data()

    # Print basic info to check if data is loaded correctly
    print(f"Training Samples: {len(X_train)}")
    print(f"Testing Samples: {len(X_test)}")

    # Check label distribution
    print(f"Train Labels Distribution: {set(y_train)}")
    print(f"Test Labels Distribution: {set(y_test)}")

    # Print first review (as tokenized numbers)
    print(f"Sample Review (First 10 words): {X_train[0][:10]}")
    print(f"Corresponding Label: {y_train[0]} (0 = Negative, 1 = Positive)")

# test_Load_imdb_data()
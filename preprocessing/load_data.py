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
    tuple: (training_set_reviews, training_set_labels, test_set_reviews, test_set_labels)
        - training_set_reviews (list of lists): Tokenized training reviews.
        - training_set_labels (list): Labels for training reviews (0 = negative, 1 = positive).
        - test_set_reviews (list of lists): Tokenized test reviews.
        - test_set_labels (list): Labels for test reviews.
    """
    imdb = tf.keras.datasets.imdb;
    (training_set_reviews, training_set_labels), (test_set_reviews, test_set_labels) = imdb.load_data(num_words=num_words);
    return training_set_reviews, training_set_labels, test_set_reviews, test_set_labels;


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
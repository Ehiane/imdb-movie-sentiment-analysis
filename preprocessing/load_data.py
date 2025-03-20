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

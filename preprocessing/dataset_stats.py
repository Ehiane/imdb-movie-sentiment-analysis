"""
Dataset Statistics Logger
=========================

This module provides insights into the IMDb dataset, helping verify 
that the data is correctly structured before training a neural network.

The function logs:
- The number of training and testing samples.
- The average length of reviews before padding.
- The distribution of sentiment labels (0 = negative, 1 = positive).

Understanding these statistics is essential for preprocessing and 
ensuring the dataset is balanced.

Functions:
----------
- log_dataset_statistics(X_train, y_train, X_test, y_test): 
  Logs dataset statistics including size, label distribution, and review lengths.

Authors:
--------
- Ehiane Oigiagbe

Created: March 19, 2025
"""

import numpy as np

def log_dataset_statistics(training_set_reviews, training_set_labels, test_set_reviews, test_set_labels):
    """
    Description:
    ------------
    Logs key statistics about the IMDb dataset to verify preprocessing correctness.

    This function helps check:
     **Total number of reviews** in the training and testing sets.
     **Average review length** before padding, to understand text distribution.
     **Label distribution** (0 = negative, 1 = positive) to ensure the dataset is balanced.
    
    Parameters:
    -----------
    training_set_reviews : list of lists
        Tokenized movie reviews for training.
    training_set_labels : list
        Sentiment labels (0 = negative, 1 = positive) for training.
    test_set_reviews : list of lists
        Tokenized movie reviews for testing.
    test_set_labels : list
        Sentiment labels (0 = negative, 1 = positive) for testing.

    Logs:
    -----
    - Number of training/testing samples.
    - Average review length before padding.
    - Label distribution (to check dataset balance).
    """

    print(f"Training Samples: {len(training_set_reviews)}")
    print(f"Testing Samples: {len(test_set_reviews)}")
    
    # Calculate average review length
    avg_train_length = np.mean([len(x) for x in training_set_reviews])
    avg_test_length = np.mean([len(x) for x in test_set_reviews])
    print(f"Average Review Length (Train): {avg_train_length:.2f}")
    print(f"Average Review Length (Test): {avg_test_length:.2f}")
    
    # Check label distribution
    train_labels, train_counts = np.unique(training_set_labels, return_counts=True)
    test_labels, test_counts = np.unique(test_set_labels, return_counts=True)

    # Format output with clear labels
    train_distribution = { "Negative Reviews (0)": train_counts[0], "Positive Reviews (1)": train_counts[1] }
    test_distribution = { "Negative Reviews (0)": test_counts[0], "Positive Reviews (1)": test_counts[1] }

    # Print label distribution
    print("Training Set Label Distribution:")
    for label, count in train_distribution.items():
        print(f"  {label}: {count}")

    print("\nTesting Set Label Distribution:")
    for label, count in test_distribution.items():
        print(f"  {label}: {count}")

import tensorflow as tf;

def decode_review(encoded_review):
    """
    Converts a tokenized IMDb review (list of word IDs) back into readable text.

    Parameters:
    -----------
    encoded_review : list
        A list of word indices representing a tokenized IMDb review.

    Returns:
    --------
    str : The decoded movie review as human-readable text.
    """
    word_index = tf.keras.datasets.imdb.get_word_index();
    
    # The dataset reserves index 1 for 'START', 2 for 'UNKNOWN', etc.
    reverse_word_index = {value: key for key, value in word_index.items()}
    
    # Decode review: replace word IDs with actual words
    decoded_review = " ".join([reverse_word_index.get(i - 3, "<?>") for i in encoded_review])
    
    return decoded_review

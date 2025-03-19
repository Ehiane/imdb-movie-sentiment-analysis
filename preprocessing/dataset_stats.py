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
    Logs important statistics about the IMDb dataset to help verify preprocessing.
    Parameters:
    -----------
    X_train : list of lists
        Tokenized movie reviews for training.
    y_train : list
        Sentiment labels (0 = negative, 1 = positive) for training.
    X_test : list of lists
        Tokenized movie reviews for testing.
    y_test : list
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
    
    print(f"Train Labels Distribution: {dict(zip(train_labels, train_counts))}")
    print(f"Test Labels Distribution: {dict(zip(test_labels, test_counts))}")


from load_data import load_imdb_data

# Load dataset using the function
X_train, y_train, X_test, y_test = load_imdb_data()

# Run the dataset statistics function to check outputs
print("\n🔹 Running Dataset Statistics Test...")
log_dataset_statistics(X_train, y_train, X_test, y_test)

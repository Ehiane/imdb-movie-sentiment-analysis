"""
Test Suite for Preprocessing Functions
======================================

This script tests all preprocessing functions together using the `unittest` framework.
It verifies:
    `__init__.py` works (modules can be imported).
    `load_imdb_data()` correctly loads dataset.
    `log_dataset_statistics()` prints dataset insights.
    `pad_reviews()` properly pads sequences.

Author:
--------
- Ehiane Oigiagbe

Created: March 19, 2025
"""

import unittest
import numpy as np
from load_data import load_imdb_data
from pad_sequences import pad_reviews
from dataset_stats import log_dataset_statistics
from dataset_stats import decode_review


class TestPreprocessing(unittest.TestCase):
    """Test Suite for Preprocessing Functions"""

    def test_init(self):
        """Test if `__init__.py` allows package imports"""

        print("\n✅ ---- Running test __init__ function ----\n");
        try:
            from load_data import load_imdb_data
            from pad_sequences import pad_reviews
            from dataset_stats import log_dataset_statistics
            print("\n✅ `__init__.py` is working! Package imports are successful.")
        except ImportError as e:
            self.fail(f"❌ `__init__.py` is NOT working! ImportError: {e}")

    def test_load_imdb_data(self):
        """Test loading IMDb dataset"""
        X_train, y_train, X_test, y_test = load_imdb_data()
        
        # Check dataset size
        self.assertEqual(len(X_train), 25000)
        self.assertEqual(len(X_test), 25000)

        # Check label distribution
        train_labels, train_counts = np.unique(y_train, return_counts=True)
        test_labels, test_counts = np.unique(y_test, return_counts=True)
        
        print("\n✅ ---- Running load imbd data function ----\n");

        print("\n✅ Training Set Label Distribution:")
        print(f"   Negative Reviews (0): {train_counts[0]}")
        print(f"   Positive Reviews (1): {train_counts[1]}")
        print("\n✅ Testing Set Label Distribution:")
        print(f"   Negative Reviews (0): {test_counts[0]}")
        print(f"   Positive Reviews (1): {test_counts[1]}")


    def test_pad_reviews(self):
        """Test padding sequences"""

        print("\n✅ ---- Running pad_reviews function ----\n");
        sample_reviews = [
            [1, 14, 22, 16],  # Short review
            [43, 530, 973, 1622, 1385, 65, 458, 4468, 66, 394],  # Medium review
            [3, 128, 12, 19, 32, 50, 70, 122, 140, 200, 300, 400, 500, 600]  # Long review
        ]
        padded_reviews = pad_reviews(sample_reviews, maxlen=10)

        print("\n✅ Original Reviews:", sample_reviews)
        print("✅ Padded Reviews:", padded_reviews)

        # Ensure all sequences have length 10
        for review in padded_reviews:
            self.assertEqual(len(review), 10)

    def test_log_dataset_stats(self):
        """Test dataset statistics logging"""

        print("\n✅ ---- Running log_dataset_stats function ----\n");
        X_train, y_train, X_test, y_test = load_imdb_data()
        print("\n✅ Running Dataset Statistics Test...")
        log_dataset_statistics(X_train, y_train, X_test, y_test)

    def test_tokenization_verification(self):
        """Test if tokenized IMDb reviews can be converted back to text"""
        print("\n✅ ---- Running tokenization verification function ----\n")
        X_train, y_train, _, _ = load_imdb_data()

        # Print a sample tokenized review
        print("\n✅ Sample Tokenized Review (First 10 words):", X_train[0][:10])

        # Decode and print the same review as text
        decoded = decode_review(X_train[0])
        print("\n✅ Decoded Review:", decoded)

        # Ensure that decoding returns a valid string
        self.assertIsInstance(decoded, str)
        self.assertGreater(len(decoded.split()), 5)  # Ensure it contains some words

if __name__ == "__main__":
    unittest.main()

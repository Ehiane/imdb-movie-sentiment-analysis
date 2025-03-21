"""
Predict Sentiment for New Movie Reviews
========================================
This script loads the trained LSTM model and predicts sentiment for new reviews.

Steps:
1. Preprocess input text.
2. Convert words to tokenized sequences.
3. Pad sequences.
4. Use trained model for prediction.

Author:
- Ehiane Oigiagbe
"""

import tensorflow as tf
import numpy as np
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
imdb = tf.keras.datasets.imdb

# Load trained model
model = tf.keras.models.load_model("models/best_lstm_model.h5")

print(model.summary());

# Function to preprocess new reviews
def preprocess_review(review, max_length=200):
    """Convert raw text review into padded sequence."""
    word_index = imdb.get_word_index()
    words = review.lower().split()  # Tokenize words
    sequence = [word_index.get(word, 2) for word in words]  # 2 = "<UNK>" for unknown words
    print("\ntokenized review:", sequence)
    padded_sequence = pad_sequences([sequence], maxlen=max_length, padding="post")
    print("\npadded Review shape:", padded_sequence.shape)
    return padded_sequence

# Example movie reviews
new_reviews = [
    "The movie was fantastic! I really enjoyed it and the actors were great.",
    "Terrible plot, bad acting, and overall boring. I regret watching this.",
    "The movie was fantastic! I really enjoyed it and the actors were great."
]

# Predict sentiment for each review
for review in new_reviews:
    processed_review = preprocess_review(review)
    prediction = model.predict(processed_review)[0][0]
    sentiment = "Positive 😊" if prediction > 0.5 else "Negative 😡"
    print(f"Review: {review}")
    print(f"Predicted Sentiment: {sentiment} (Confidence: {prediction:.4f})\n")

"""
LSTM Model Definition
======================
This script defines the Long Short-Term Memory (LSTM) model used
for classifying IMDb movie reviews as positive or negative.

It initializes an embedding layer with GloVe embeddings, followed by
two LSTM layers and a final dense layer for classification.

Components:
------------
- Uses GloVe word embeddings for better understanding of words.
- Employs dropout layers to prevent overfitting.
- Outputs a probability score (0 = negative, 1 = positive).

Authors:
---------
- Ehiane Oigiagbe

Created: March 19, 2025
"""

import tensorflow as tf

# Add project root to sys.path
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from preprocessing.embedding import create_embedding_matrix

# Access Keras components directly from `tf.keras`
Sequential = tf.keras.models.Sequential
Embedding = tf.keras.layers.Embedding
LSTM = tf.keras.layers.LSTM
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout

# Define constants
VOCAB_SIZE = 10000
EMBEDDING_DIM = 100
MAX_LENGTH = 200
GLOVE_PATH = "../data/glove.6B.100d.txt"

# Create embedding matrix
embedding_matrix = create_embedding_matrix(VOCAB_SIZE, EMBEDDING_DIM, GLOVE_PATH)

# Define LSTM Model
model = Sequential([
    Embedding(
        input_dim=VOCAB_SIZE, 
        output_dim=EMBEDDING_DIM, 
        input_length=MAX_LENGTH,
        weights=[embedding_matrix],  # Load pretrained embeddings
        trainable=False  # Freeze embeddings (set to True for fine-tuning)
    ),
    LSTM(128, return_sequences=True),
    Dropout(0.5),
    LSTM(64),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

print("✅ Model architecture ready with GloVe embeddings!")

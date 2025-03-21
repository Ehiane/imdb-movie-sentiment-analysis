"""
LSTM Model Training Script
============================
This script prepares and trains an LSTM model for sentiment analysis
on the IMDb movie review dataset. It first loads the dataset, ensures
all reviews are of equal length using padding, then trains the model.

Finally, the trained model is saved for future predictions.

Steps:
------
1. Load IMDb dataset.
2. Apply padding to standardize review lengths.
3. Train an LSTM model using word embeddings.
4. Save the trained model.

Authors:
---------
- Ehiane Oigiagbe

Created: March 19, 2025
"""

import tensorflow as tf
from lstm_model import model


# Dynamically access TensorFlow components
imdb = tf.keras.datasets.imdb
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
EarlyStopping = tf.keras.callbacks.EarlyStopping;
ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint;

# Load IMDb dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=10000)

# Pad sequences (ensure all reviews have the same length)
X_train = pad_sequences(X_train, maxlen=200)
X_test = pad_sequences(X_test, maxlen=200)


# define early stopping to stop training if validation loss doesn't improve
early_stopping = EarlyStopping(
                    monitor='val_loss', 
                    patience=3, 
                    restore_best_weights= True);

# define model checkpointing to save the best model 
checkpoint_path = "models/best_lstm_model.h5";
checkpoint = ModelCheckpoint(
                filepath=checkpoint_path, 
                save_best_only=True, 
                monitor= "val_accuracy", 
                mode="max");


# Train the model with callbacks
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=64,
    validation_data=(X_test, y_test),
    callbacks = [early_stopping, checkpoint]
)

import json

# Convert history to JSON and save
history_dict = history.history
with open("models/training_history.json", "w") as f:
    json.dump(history_dict, f)

# Save the model
model.save("models/lstm_glove_imdb.h5")
print("✅ Model training complete &  training history saved!")


# to run this: python .\train_model.py
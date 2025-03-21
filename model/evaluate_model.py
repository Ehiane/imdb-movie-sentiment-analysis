"""
Evaluate Model Performance
===========================
This script visualizes training progress by plotting accuracy and loss curves.

Steps:
1. Load training history.
2. Plot training vs. validation accuracy.
3. Plot training vs. validation loss.

Author:
- Ehiane Oigiagbe
"""

import matplotlib.pyplot as plt
import json

# Load training history (if saved)
with open("models/training_history.json", "r") as f:
    history = json.load(f)

# Extract metrics
epochs = range(1, len(history["accuracy"]) + 1)

# Plot Accuracy
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, history["accuracy"], label="Training Accuracy")
plt.plot(epochs, history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(epochs, history["loss"], label="Training Loss")
plt.plot(epochs, history["val_loss"], label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()

# Show plots
plt.tight_layout()
plt.show()

# python .\evaluate_model.py
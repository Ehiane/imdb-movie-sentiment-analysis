import tensorflow as tf
from lstm_model import model

def test_model_structure():
    """Ensure model has correct layer types"""
    layers = [type(layer).__name__ for layer in model.layers]
    expected_layers = ['Embedding', 'LSTM', 'Dropout', 'LSTM', 'Dropout', 'Dense']
    assert layers == expected_layers, f"Model layers incorrect: {layers}"

def test_model_compilation():
    """Ensure model is compiled correctly"""
    assert model.loss == "binary_crossentropy"
    assert isinstance(model.optimizer, tf.keras.optimizers.Adam)

def test_model_training():
    """Ensure model can run a single batch without crashing"""
    import numpy as np
    X_dummy = np.random.randint(0, 10000, (10, 200))  # Fake 10 reviews
    y_dummy = np.random.randint(0, 2, (10,))  # Fake labels
    history = model.fit(X_dummy, y_dummy, epochs=1, batch_size=2, verbose=0)
    assert history is not None

if __name__ == "__main__":
    test_model_structure()
    test_model_compilation()
    test_model_training()
    print("✅ All model tests passed!")

# to run this use : pytest test_model.py
"""
Description:
------------
    Padding Sequences for LSTM Input
    ================================

    This module ensures that all movie reviews are of uniform length by padding 
    or truncating them to a fixed size. This is necessary for feeding data into 
    an LSTM model, which requires equal-length sequences.

Functions:
----------
- pad_reviews(sequences, maxlen=200): Pads or truncates sequences to a fixed length.

Authors:
--------
- Ehiane Oigiagbe

Created: March 19, 2025
"""

import tensorflow as tf

def pad_reviews(sequences, maxlen=200):
    """
    Description:
    ------------
        LSTMs and Neural Networks require input of the same size, therefore this function, 
        Pads or truncates sequences to ensure uniform input size:
            If a review is shorter than 10 words, it adds zeros (0) at the end.
            If a review is longer than 10 words, it cuts off extra words at the end.

    Parameters:
        sequences (list of lists): Tokenized reviews.
        maxlen (int): Maximum sequence length.

    Returns:
        numpy.ndarray: Padded sequences.
    """
    return tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=maxlen, padding='post', truncating='post');

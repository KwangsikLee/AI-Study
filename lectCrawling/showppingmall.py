import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
from collections import Counter
from konlpy.tag import Mecab
from sklearn.model_selection import train_test_split


import tensorflow as tf
#from tensorflow.keras import preprocessing
# from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Sample text data
text_data = ["This is a sample sentence.", "Another sentence for demonstration."]

# Create a TextVectorization layer
vectorize_layer = tf.keras.layers.TextVectorization(
    max_tokens=None,  # Or specify a maximum vocabulary size
    output_mode='int',
    output_sequence_length=None # Or specify a fixed sequence length
)

# Adapt the layer to your text data to build the vocabulary
vectorize_layer.adapt(text_data)

# Apply the layer to new text data
vectorized_text = vectorize_layer(tf.constant(text_data))

print(vectorized_text)

# urllib.request.urlretrieve("https://raw.githubusercontent.com/bab2min/corpus/master/sentiment/naver_shopping.txt", filename="ratings_total.txt")
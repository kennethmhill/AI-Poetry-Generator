import numpy as np  
import pandas as pd 
import matplotlib.pyplot as plt 
import tensorflow.keras.utils as ku  
from tensorflow.keras.preprocessing.sequence import pad_sequences 
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, Input 
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.optimizers import Adam 
from tensorflow.keras import regularizers

# read in poem.txt, which contains The Raven by Edgar Allen Poe
data = open('poem.txt', encoding="utf8").read() 
# generate the corpus, split text by line, convert to lowercase
corpus = data.lower().split("\n") 
# print(corpus[:10]) # print first 10 lines

# fit the Tokenizer on the Corpus
tokenizer = Tokenizer() # a Tokenizer converts words into unique integer indices
tokenizer.fit_on_texts(corpus) 
  
# .word_index creates a dictionary mapping each unique word to an integer.
total_words = len(tokenizer.word_index) 
print("Total Words:", total_words) # count of unique words in the corpus

# convert the text into embeddings 
# Each line in the poem is converted into a list of integers (token_list) using the tokenizer.
input_sequences = [] 
for line in corpus: 
    token_list = tokenizer.texts_to_sequences([line])[0] 
    
    # n-gram sequences are generated. For example, [10,20,30] breaks into [10,20] and [10]
    # generates multiple training samples for the model
    for i in range(1, len(token_list)): 
        n_gram_sequence = token_list[:i+1] 
        input_sequences.append(n_gram_sequence) 

# all sequences are padded to the same length (max_sequence_len) by prepending zeros
max_sequence_len = max([len(x) for x in input_sequences]) 
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')) 

# Input sequences are split into:
# Predictors: All tokens except the last one.
# Label: The last token in each sequence.
predictors, label = input_sequences[:, :-1], input_sequences[:, -1] 

# Labels are one-hot encoded using to_categorical.
label = ku.to_categorical(label, num_classes=total_words+1) 
print(input_sequences)

# define the bi-directional LSTM model
model = Sequential() 
model.add(Input(shape=(max_sequence_len,)))  # Define the input layer

#  Converts word indices into dense vectors of fixed size (100-dimensional).
model.add(Embedding(input_dim=total_words + 1, output_dim=100))

# Captures both past and future context in sequences
model.add(Bidirectional(LSTM(150, return_sequences=True)))

# Prevents overfitting by randomly deactivating 20% of neurons during training
model.add(Dropout(0.2))

# Learns temporal dependencies in the data
model.add(LSTM(100))

# First dense layer reduces dimensions with ReLU activation
model.add(Dense((total_words + 1) // 2, activation='relu', kernel_regularizer=regularizers.l2(0.01)))

# Final dense layer outputs probabilities for each word in the vocabulary using a softmax activation
model.add(Dense(total_words + 1, activation='softmax'))

# uses categorical crossentropy loss for multi-class classification and the Adam optimizer for efficient gradient descent.
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Print the model summary
print(model.summary())

# train the model
history = model.fit(predictors, label, epochs=150, verbose=1)
# model is trained on the input data for 150 epochs.
# predictors are the input sequences, and label represents the target words.
# During training, the dataset is divided into smaller chunks called batches. The model processes each 
# batch sequentially to update its parameters (e.g., weights in a neural network). Once all batches in 
# the dataset are processed, one epoch is complete.
# Each epoch gives the model a chance to adjust its parameters to reduce the loss function.
# Multiple epochs allow the model to learn patterns in the data better.

# use the model to generate text (next 25 words) based on a seed 
seed_text = "The raven"
next_words = 25
ouptut_text = "" 
  
for _ in range(next_words): 
    token_list = tokenizer.texts_to_sequences([seed_text])[0] 
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre') 
    predicted = np.argmax(model.predict(token_list,  verbose=0), axis=-1) 
    output_word = "" 
      
    for word, index in tokenizer.word_index.items(): 
        if index == predicted: 
            output_word = word 
            break
              
    seed_text += " " + output_word 
      
print(seed_text) 
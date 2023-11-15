#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
print(tf.__version__)


# In[2]:


import string
import requests


# In[3]:


response = requests.get('https://www.gutenberg.org/cache/epub/18993/pg18993.txt')


# In[4]:


response.text


# In[5]:


data = response.text.replace('\r', "").replace("\\'", "'").strip()
data = data.split('\n')


# In[6]:


data[0]


# In[7]:


data[737]


# In[8]:


data = data[737:]


# In[9]:


data[0]


# In[10]:


len(data)


# In[11]:


data = " ".join(data)


# In[12]:


data


# In[13]:


def clean_text(doc):
  tokens = doc.split()
  table = str.maketrans('','',string.punctuation)
  tokens = [(w.translate(table)) for w in tokens]
  tokens = [word for word in tokens if word.isalpha()]
  tokens = [word.lower() for word in tokens]
  return tokens


# In[14]:


tokens = clean_text(data)


# In[15]:


print(tokens[:50])


# In[16]:


len(tokens)


# In[17]:


len(set(tokens))


# In[18]:


length = 50 +1
lines = []

for i in range(length, len(tokens)):
  seq = tokens[i-length: i]
  line = ' '.join(seq)
  lines.append(line)
  if i > 200000:
    break


# In[19]:


print(len(lines))


# In[20]:


lines[0]


# In[21]:


lines[1]


# In[22]:


import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences


# In[23]:


tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)
sequences = tokenizer.texts_to_sequences(lines)


# In[24]:


sequences = np.array(sequences)


# In[25]:


sequences


# In[26]:


x = sequences[:, :-1]
y = sequences[:, -1]


# In[27]:


x[0]


# In[28]:


y[0]


# In[29]:


tokenizer.word_index


# In[30]:


len(tokenizer.word_index)


# In[31]:


vocab_size = len(tokenizer.word_index) + 1


# In[32]:


vocab_size


# In[33]:


len(set(tokens))


# In[34]:


y = to_categorical(y, num_classes=vocab_size)


# In[35]:


x.shape[1]


# In[36]:


seq_length = x.shape[1]


# In[37]:


seq_length


# In[38]:


model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=50, input_length=seq_length))
model.add(LSTM(units=100, return_sequences=True))
model.add(LSTM(units=100))
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=vocab_size, activation='softmax'))


# In[39]:


model.summary()


# In[40]:


model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


# In[41]:


model.fit(x,y,batch_size=256,epochs=100)


# In[42]:


lines[10299]


# In[43]:


seed_text = lines[10299]


# In[44]:


def generate_text_seq(model, tokenizer, text_seq_length, seed_text, n_words):
    text = []
    
    for _ in range(n_words):
        encoded = tokenizer.texts_to_sequences([seed_text])[0]
        encoded = pad_sequences([encoded], maxlen = text_seq_length, truncating='pre')
        
        y_predict = model.predict(encoded)[0]
        
        predicted_word_idx = np.argmax(y_predict)
        
        predicted_word = ''
        for word, index in tokenizer.word_index.items():
            if index == predicted_word_idx:
                predicted_word = word
                break
        seed_text = seed_text + ' ' + predicted_word
        text.append(predicted_word)
    
    return ' '.join(text)


# In[45]:


generate_text_seq(model, tokenizer, seq_length, seed_text, 10)


# In[46]:


seed_text = 'The german initiated the war because they are'


# In[47]:


generate_text_seq(model, tokenizer, seq_length, seed_text, 10)


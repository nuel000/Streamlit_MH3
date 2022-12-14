# Importing modules for data science and visualization
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import nltk
nltk.download('averaged_perceptron_tagger')

import nlppreprocess
# NLP Libraries
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from nlppreprocess import NLP
from nltk import pos_tag
import pickle
import json

# reading in the dataset
train = pd.read_csv("train.csv")

# Data cleaning for furthur sentiment analysis

def clean_tweet(message):
    
    message = re.sub(r'^RT ','', re.sub(r'https://t.co/\w+', '', message).strip()) 

    emojis = re.compile("["
                           u"\U0001F600-\U0001F64F"  # removes emoticons,
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)

    message = emojis.sub(r'', message)
    # Removing puctuation
    punctuation = re.compile("[.;:!\'’‘“”?,\"()\[\]]")
    tweet = punctuation.sub("", message) 


    # Removing stopwords
    nlp_for_stopwords = NLP(replace_words=True, remove_stopwords=True, 
                            remove_numbers=True, remove_punctuations=False) 
    tweet = nlp_for_stopwords.process(tweet) # removes stop words 
    
    #tokenization
    tweet = tweet.split() 

    # Part of Speech
    pos = pos_tag(tweet)
 
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tweet = ' '.join([lemmatizer.lemmatize(word, po[0]) 
                      if (po[0] in ['n', 'r', 'v', 'a'] and word[0] != '@') else word for word, po in pos])


    return tweet

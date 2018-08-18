# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 18:50:21 2018

@author: onkar
"""

import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
example_sent = "the biggest risk is not taking any risk...."
stop_words = set(stopwords.words('english'))
word_tokens = word_tokenize(example_sent)
filtered_sentence = [x for x in word_tokens if not x in stop_words]
filtered_sentence = []
for x in word_tokens:
 if x not in stop_words:
     filtered_sentence.append(x)
print("\n\nTHESE ARE WORD TOKENS __________________")
print(word_tokens)
print("\n\nFILTERED SENTENCES ________________________")
print(filtered_sentence)
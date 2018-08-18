# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 18:55:51 2018

@author: onkar
"""

from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
words = ['The', 'biggest', 'risk', 'taking', 'risk']
ps = PorterStemmer()
for word in words:
 print(word+"\t stem word:-\t"+ps.stem(word))
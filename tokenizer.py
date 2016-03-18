#!/usr/bin/env python3
import re
import sys
import csv
import math
from collections import Counter

stopwords = []
with open('stopwords-tr.csv', 'r') as f:
	# Stop words are taken from http://www.turkceogretimi.com/Genel-Konular/article/541-turkce-etkisiz-kelimeler-stop-words-listesi-11/35
	stopwords = [word for line in list(csv.reader(f)) for word in line]

# Matches words by stripping all punctuation around it. Also strips the outer part of an apostrophe.
token_re = re.compile(R"[!\"#$%&'()*+,\-./:;<=>?@[\]^_`{|}~]*([\w-]+)(?:'\w*)*[!\"#$%&'()*+,\-./:;<=>?@[\]^_`{|}~]*", re.U)
# Matches the end of the sentences by looking for [.?:!] characters. Do note that dots are not matched if they are 
# surrounded by numbers (ex: 5.44).
sentence_re = re.compile(R"(?:(?<=\d)\.(?!\d))|(?:(?<!\d)\.(?=\d))|(?:(?<!\d)\.(?!\d))|\.\.+|[!?:]+", re.U)

def is_number(s):
		try:
			float(s)
			return True
		except ValueError:
			return False

def tokenize(line):
		""" Tokenizes the given line by splitting it from whitespaces and punctuations
		that are commonly found at the end of sentences. Does not include
		tokens of length 0. All tokens are transformed to lowercase unless the whole
		token is uppercase.

		Returns a list of tokens.
		"""
		m = re.findall(token_re, line.strip())
		if m == None:
			return []
		return [ (lambda tok: tok if tok.isupper() else tok.lower())(token) for token in m if not(len(token) == 0 or token in stopwords) ]

def half_round(number):
	return round(number * 2) / 2

class Tokenizer:
	""" Tokenizer for Turkish
	There are four use cases:
	1. Token stream through has_next() and next_token() methods.
	2. List of all word tokens through bag_of_words() method.
	3. List of all character n-grams through bag_of_char_ngrams() method.
	4. List of features through features() method.
	"""
	def __init__(self, path = None):
		""" Initializes the tokenizer with the given text file path.
		Note that the given files are assumed to be of Windows-1254 (Turkish)
		encoding. Reads the whole file and splits it into sentences.
		"""
		if not path == None:
			file = open(path, 'r', encoding = 'cp1254')
			lines = file.readlines()
			file.close()
			self.original = " ".join(lines)
			self.sentences = [sentence.strip() for sentence in re.split(sentence_re ,self.original) if not sentence.strip() == '']
		else:
			self.sentences = []
		self.tokens = {}
		self.line = []

	def append_sentences(self, sentences):
		""" Appends sentences to already existing sentences. """
		self.sentences.extend(sentences)

	def bag_of_words(self):
		""" Returns the bag of words representation of the file. """
		return Counter([token for sentence in self.sentences for token in tokenize(sentence)])

	def char_ngrams(self, token, n):
		""" Returns the list of char n-grams from a given token. """
		num_ngrams = len(token)-n+1
		lis = []
		for i in range(num_ngrams):
			lis.append(token[i:i+n])
		return lis

	def bag_of_char_ngrams(self, n):
		""" Returns the bag of char n-grams representation of the file. """
		return Counter([ngram for sentence in self.sentences for token in tokenize(sentence) for ngram in self.char_ngrams(token,n)])

	def features(self):
		""" Returns features extracted from sentences.
		The features is a 8-tuple of the following kind:
		(number of sentences, list of number of words in a sentence, list of word lengths, list of number of commas in a sentence,
			list of number of exclamation marks in a sentence, list of number of question marks in a sentence, list of number 
			of periods in a sentence, average number of unique words per word)
		"""
		words_in_sentences = []
		word_len = []
		commas_in_sentences = []
		total_excl = self.original.count('!')
		excl = [] + [1] * total_excl + [0] * max(0,len(self.sentences)-total_excl)
		total_ques = self.original.count('?')
		ques = [] + [1] * total_ques + [0] * max(0,len(self.sentences)-total_ques)
		total_period = self.original.count('.')
		period = [] + [1] * total_period + [0] * max(0,len(self.sentences)-total_period)
		unique_words = set([])
		word_count = 0
		for sentence in self.sentences:
			commas_in_sentences.append(sentence.count(','))
			words = tokenize(sentence)
			words_in_sentences.append(len(words))
			for word in words:
				word_len.append(len(word))
				unique_words.add(word)
				word_count+=1
		return len(self.sentences), words_in_sentences, word_len, commas_in_sentences, excl, ques, period, len(unique_words)/word_count

	def has_next(self):
		""" Returns True if the token stream has any tokens left.

		Handles the acquiring of new tokens from the given token stream (file).
		Note that the sentences are exhausted using this method.
		"""
		if len(self.line) > 0: return True
		if len(self.sentences) == 0: return False
		else:
			line = self.sentences.pop(0)
			self.line = tokenize(line)
			return self.has_next()

	def next_token(self):
		""" Returns the next token in the token stream. """
		if len(self.line) > 0:
			token = self.line.pop(0)
			if token in self.tokens:
				self.tokens[token] = self.tokens[token] + 1
			else:
				self.tokens[token] = 1
			return token
		else: # Should only happen at EOF
			return ''

if __name__ == '__main__':
	""" Accepts one mandatory argument: text path """
	if len(sys.argv) < 2:
		print('Please enter the directory to load the text from.')
	else:
		t = Tokenizer(sys.argv[1])
		while t.has_next():
			token = t.next_token()
			print(token)
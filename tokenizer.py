#!/usr/bin/env python3
import re
import sys
import csv
import math

stopwords = []
with open('stopwords-tr.csv', 'r') as f:
	# Stop words are taken from http://www.turkceogretimi.com/Genel-Konular/article/541-turkce-etkisiz-kelimeler-stop-words-listesi-11/35
	stopwords = [word for line in list(csv.reader(f)) for word in line]

# Matches words by stripping all punctuation around it. Also strips the outer part of an apostrophe.
token_re = re.compile(R"[!\"#$%&'()*+,\-./:;<=>?@[\]^_`{|}~]*([\w-]+)(?:'\w*)*[!\"#$%&'()*+,\-./:;<=>?@[\]^_`{|}~]*", re.U)

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

class Tokenizer:
	""" Tokenizer for Turkish
	There are two use cases:
	1. Token stream through has_next() and next_token() methods.
	2. List of all tokens through bag_of_words() method
	"""
	def __init__(self, path = None):
		""" Initializes the tokenizer with the given text file path.
		Note that the given files are assumed to be of Windows-1254 (Turkish)
		encoding.
		"""
		if not path == None:
			file = open(path, 'r', encoding = 'cp1254')
			lines = file.readlines()
			file.close()
			joined = " ".join(lines)
			self.sentences = [sentence.strip() for sentence in re.split("(?:(?<=\d)\.(?!\d))|(?:(?<!\d)\.(?=\d))|(?:(?<!\d)\.(?!\d))|\.\.+|[!?:]+",joined) if not sentence.strip() == '']
		else:
			self.sentences = []
		self.tokens = {}
		self.line = []

	def append_sentences(self, sentences):
		""" Appends sentences to already existing sentences. """
		self.sentences.extend(sentences)

	def bag_of_words(self):
		""" Returns the bag of words representation of the file. """
		return [token for sentence in self.sentences for token in tokenize(sentence)]

	def has_next(self):
		""" Returns True if the token stream has any tokens left.

		Handles the acquiring of new tokens from the given token stream (file). 
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
#!/usr/bin/env python3
import re
import sys

class Tokenizer:
	def __init__(self, path, count = False):
		"""
		Initializes the tokenizer with the given text file path. If set,
		count variable enables the built in frequency collection about
		token counts.
		Note that the given files are assumed to be of Windows-1254 (Turkish)
		encoding.
		"""
		self.file = open(path, 'r', encoding = 'cp1254')
		self.tokens = {}
		self.line = []

	def has_next(self):
		"""
		Handles the acquiring of new tokens from the given token stream (file).
		Returns True if the token stream has any tokens left. 
		"""
		if len(self.line) > 0:
			return True
		else:
			line = self.file.readline()
			if line == '': #EOF
				self.file.close()
				return False
			else:
				self.line = self.tokenize(line)
				return self.has_next()

	def tokenize(self, line):
		"""
		Tokenizes the given line by splitting it from whitespaces and punctuations
		that are commonly found at the end of sentences. Does not include
		tokens of length 0. All tokens are transformed to lowercase unless the whole
		token is uppercase.

		Returns a list of tokens.
		"""
		return [ (lambda tok: tok if tok.isupper() else tok.lower())(token) for token in re.split('[\s\.()\-"“!;,\'\?:]+', line.strip()) if not len(token) == 0 ]

	def next_token(self):
		"""
		Returns the next token.
		"""
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
	"""
	Accepts one mandatory argument: text path
	"""
	if len(sys.argv) < 2:
		print('Please enter the directory to load the text from.')
	else:
		t = Tokenizer(sys.argv[1])
		while t.has_next():
			token = t.next_token()
			print(token)
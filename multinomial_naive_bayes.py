#!/usr/bin/env python3
from preprocessor import Preprocessor
from tokenizer import Tokenizer
import math
import sys

class MultinomialNaiveBayes:
	def __init__(self):
		self.class_data = { }
		self.total_documents = 0
		self.vocabulary = set([])

	def add_class(self, class_name):
		""" Adds a class with the given class name.
		This method should be called for each class in the training set.
		"""
		self.class_data[class_name] = { 'num_documents' : 0, 'total_word_count': 0, 'words' : {} }

	def add_document(self, class_name, amount = 1):
		""" Adds a document with the given class.
		This method should be called until the total amount matches the number of documents
		in the training set for each class.
		"""
		self.total_documents += amount
		self.class_data[class_name]['num_documents'] += amount

	def add_word(self, class_name, word):
		""" Convenience method for adding a single word to the given class bag """
		self.vocabulary.add(word)
		self.class_data[class_name]['total_word_count'] += 1
		if word in self.class_data[class_name]['words']:
			self.class_data[class_name]['words'][word] += 1
		else:
			self.class_data[class_name]['words'][word] = 1

	def most_probable_class(self, words, alpha = 1):
		""" Returns the class with the highest log probability, given a bag of words.
		Alpha is the Laplace smoothing parameter.
		"""
		best = -float("inf"), ''
		for class_name in self.class_data.keys():
			prob = self.get_class_log_probability(class_name, words, alpha)
			if prob > best[0]:
				best = prob, class_name
		return best[1]

	def get_class_log_probability(self, class_name, words = [], alpha = 1):
		""" Gets the log probability of a bag of words being in a certain class.
		Alpha is the Laplace smoothing parameter.
		"""
		prob = math.log(self.class_data[class_name]['num_documents']  / self.total_documents)
		for word in words:
			prob += self.get_word_log_probability(class_name, word, alpha = alpha)
		return prob

	def get_word_log_probability(self, class_name, word, alpha = 1):
		""" Gets the log probability of a word being in a certain class.
		Uses Laplace smoothing with parameter alpha.
		"""
		word_count = 0
		if word in self.class_data[class_name]['words']:
			word_count = self.class_data[class_name]['words'][word]
		return math.log((word_count + alpha) / (self.class_data[class_name]['total_word_count'] + alpha * len(self.vocabulary)))

if __name__ == '__main__':
	""" Performs the basic example from the lecture notes. """
	bayes = MultinomialNaiveBayes()
	bayes.add_class('china')
	bayes.add_class('japan')
	china = 'Chinese Beijing Chinese Chinese Chinese Shanghai Chinese Macao'.split(' ')
	japan = 'Tokyo Japan Chinese'.split(' ')
	bayes.add_document('china', amount = 3)
	bayes.add_document('japan', amount = 1)
	for c in china:
		bayes.add_word('china', c)
	for j in japan:
		bayes.add_word('japan', j)

	print(bayes.most_probable_class('Chinese Chinese Chinese Tokyo Japan'.split(' ')))
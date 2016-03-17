#!/usr/bin/env python3
from preprocessor import Preprocessor
from tokenizer import Tokenizer
import numpy as np
import math
import sys

class MultinomialNaiveBayes:
	def __init__(self, classes):
		self.classes = dict(zip(classes, range(0, len(classes))))
		self.class_indices = dict([reversed(i) for i in self.classes.items()])
		self.class_data = { }
		for class_name in classes:
			self.class_data[class_name] = { 'num_documents' : 0, 'total_word_count': 0, 'words' : {} }
		self.total_documents = 0
		self.vocabulary = set([])

	def get_classes(self):
		""" Convenience method for getting classes and their indexes. """
		return self.classes

	def add_documents(self, class_name, amount = 1):
		""" Adds a number of documents to the given class.
		This method should be called until the total amount matches the number of documents
		in the training set for each class.
		"""
		self.total_documents += amount
		self.class_data[class_name]['num_documents'] += amount

	def add_word(self, class_name, word, amount = 1):
		""" Convenience method for adding a single word to the given class bag """
		self.vocabulary.add(word)
		self.class_data[class_name]['total_word_count'] += amount
		if word in self.class_data[class_name]['words']:
			self.class_data[class_name]['words'][word] += amount
		else:
			self.class_data[class_name]['words'][word] = amount

	def most_probable_class(self, words, alpha = 1):
		""" Returns the class with the highest log probability, given a bag of words.
		Alpha is the Laplace smoothing parameter.
		"""	
		return self.class_indices[np.argmax(self.class_log_probabilities(words, alpha))]

	def class_log_probabilities(self, words, alpha = 1, normalize = True):
		""" Returns the log probabilities of all classes in a vector,
		where each index corresponds to the class given by self.class_indices[index].
		Alpha is the Laplace smoothing parameter. If normalize is True, all log probabilities
		are normalized to [0,1] interval (1 being the highest log probability) while retaining their orderings.
		"""
		probs = np.array([ (lambda x: self.get_class_log_probability(self.class_indices[x], words, alpha = alpha))(ind) for ind in self.class_indices.keys()])
		if not normalize: return probs
		pos_probs = probs-np.min(probs)
		return pos_probs / np.max(pos_probs)

	def get_class_log_probability(self, class_name, words = [], alpha = 1):
		""" Gets the log probability of a bag of words being in a certain class.
		Alpha is the Laplace smoothing parameter.
		"""
		prob = math.log(self.class_data[class_name]['num_documents']  / self.total_documents)
		return prob + np.sum([ (lambda x: self.get_word_log_probability(class_name, x, alpha = alpha))(word) for word in words ])

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
	bayes = MultinomialNaiveBayes(['china','japan'])
	china = 'Chinese Beijing Chinese Chinese Chinese Shanghai Chinese Macao'.split(' ')
	japan = 'Tokyo Japan Chinese'.split(' ')
	bayes.add_documents('china', amount = 3)
	bayes.add_documents('japan', amount = 1)
	for c in china:
		bayes.add_word('china', c)
	for j in japan:
		bayes.add_word('japan', j)

	print(bayes.most_probable_class('Chinese Chinese Chinese Tokyo Japan'.split(' ')))
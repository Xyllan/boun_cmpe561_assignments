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
		"""
		Adds a class with the given class name.
		This method should be called for each class in the training set.
		"""
		self.class_data[class_name] = { 'num_documents' : 0, 'total_word_count': 0, 'words' : {} }

	def add_document(self, class_name):
		"""
		Adds a document with the given class.
		This method should be called for each document in the training set.
		"""
		self.total_documents += 1
		self.class_data[class_name]['num_documents'] += 1

	def add_word(self, class_name, word):
		"""
		Convenience method for adding a single word to the given class bag
		"""
		self.vocabulary.add(word)
		self.class_data[class_name]['total_word_count'] += 1
		if word in self.class_data[class_name]['words']:
			self.class_data[class_name]['words'][word] += 1
		else:
			self.class_data[class_name]['words'][word] = 1

	def most_probable_class(self, words, alpha = 1):
		"""
		Returns the class with the highest log probability, given a bag of words.
		Alpha is the Laplace smoothing parameter.
		"""
		best = -float("inf"), ''
		for class_name in self.class_data.keys():
			prob = self.get_class_log_probability(class_name, words, alpha)
			if prob > best[0]:
				best = prob, class_name
		return best[1]

	def get_class_log_probability(self, class_name, words = [], alpha = 1):
		"""
		Gets the log probability of a bag of words being in a certain class.
		Alpha is the Laplace smoothing parameter.
		"""
		prob = math.log(self.class_data[class_name]['num_documents']  / self.total_documents)
		for word in words:
			prob += bayes.get_word_log_probability(class_name, word, alpha = alpha)
		return prob

	def get_word_log_probability(self, class_name, word, alpha = 1):
		"""
		Gets the log probability of a word being in a certain class.
		Uses Laplace smoothing with parameter alpha.
		"""
		word_count = 0
		if word in self.class_data[class_name]['words']:
			word_count = self.class_data[class_name]['words'][word]
		return math.log((word_count + alpha) / (self.class_data[class_name]['total_word_count'] + alpha * len(self.vocabulary)))

if __name__ == '__main__':
	"""
	Accepts two arguments, one mandatory (authors directory) and one optional
	(seed for random number generation)
	"""
	if len(sys.argv) < 2:
		print('Please enter the directory to load authors from.')
	else:
		seed = 1232
		if len(sys.argv) > 2:
			seed = sys.argv[2]

		p = Preprocessor(seed, sys.argv[1])
		p.organize()
		bayes = MultinomialNaiveBayes()
		authors = p.get_authors()
		for author in authors:
			bayes.add_class(author)
			for data in p.training_data(author):
				bayes.add_document(author)
				t = Tokenizer(p.file_path(author,data))
				while t.has_next():
					token = t.next_token()
					bayes.add_word(author, token)

		t = Tokenizer('69yazar/raw_texts__test/abbasGuclu/4.txt')
		lis = []
		while t.has_next():
			lis.append(t.next_token())
		print(bayes.most_probable_class(lis))
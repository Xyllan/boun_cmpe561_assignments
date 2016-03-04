#!/usr/bin/env python3
from preprocessor import Preprocessor
from tokenizer import Tokenizer
from multinomial_naive_bayes import MultinomialNaiveBayes
import numpy as np
import os
import getopt
import sys

def f_score(precision, recall, beta = 1):
	return ((beta ** 2 + 1) * precision * recall) / ((beta ** 2) * precision + recall)

def div0(a, b):
	"""
	Elementwise divide two numpy arrays, with divide by zero equaling 0.
	div0( [-1, 0, 1], 0 ) would therefore return [0, 0, 0]
	This is done avoid high precision values for the case where 
	a class is not marked with neither false nor true positive.
	"""
	with np.errstate(divide='ignore', invalid='ignore'):
		c = np.true_divide(a, b)
		c[ ~np.isfinite(c)] = 0  # -inf inf NaN
	return c

def test_authors(p):
	bayes = MultinomialNaiveBayes()
	authors = p.get_authors()
		
	# Train the bayes classifier for each training data
	for author in authors:
		bayes.add_class(author)
		for data in p.training_data(author):
			bayes.add_document(author)

			# Tokenize and add tokens to the bag of words
			t = Tokenizer(p.file_path(author,data))
			while t.has_next():
				bayes.add_word(author, t.next_token())

	tester = Tester(authors)
	# Check the bayes classifier for each 
	for author in authors:
		for data in p.test_data(author):

			# Tokenize and add tokens to the bag of words
			t = Tokenizer(p.file_path(author,data, training_data = False))
			bagOfWords = []
			while t.has_next():
				bagOfWords.append(t.next_token())

			#TODO collect statistics 
			class_predicted = bayes.most_probable_class(bagOfWords)
			tester.add_stat(class_predicted, author)
		
	tester.print_scores()

class Tester:
	def __init__(self, classes):
		cls_len = len(classes)
		index = 0
		self.classes = dict(zip(classes, range(0, cls_len)))
		self.stats = np.zeros((cls_len, cls_len), dtype=np.int)

	def add_stat(self, predicted_class, real_class):
		self.stats[self.classes[real_class],self.classes[predicted_class]] += 1

	def microavg_precision(self):
		return np.trace(self.stats) / np.sum(self.stats)

	def microavg_recall(self):
		return np.trace(self.stats) / np.sum(self.stats)

	def macroavg_precision(self):
		return np.nanmean(div0(np.diagonal(self.stats), np.sum(self.stats, axis=0)))

	def macroavg_recall(self):
		return np.nanmean(div0(np.diagonal(self.stats), np.sum(self.stats, axis=1)))

	def print_scores(self):
		micro_prec = self.microavg_precision()
		micro_rec = self.microavg_recall()
		micro_f = f_score(micro_prec, micro_rec)
		macro_prec = self.macroavg_precision()
		macro_rec = self.macroavg_recall()
		macro_f = f_score(macro_prec, macro_rec)
		print('Micro-averaged scores:')
		print('Precision:',micro_prec)
		print('Recall:', micro_rec)
		print('F-score (beta=1):', micro_f)
		print('Macro-averaged scores:')
		print('Precision:',macro_prec)
		print('Recall:', macro_rec)
		print('F-score (beta=1):', macro_f)

if __name__ == '__main__':
	"""
	This program normally accepts two arguments: the directory of the training set and
	the directory of the test set. If used, the -p option will make the program do the 
	training/test set preprocessing with the given outer directory. If -p is used,
	the -s option followed by a number can also be used to set the random seed
	for test data shuffling.
	"""
	seed = None
	prep = False
	argv = []
	p = Preprocessor()
	try:
		optlist, argv = getopt.getopt(sys.argv[1:], 'ps:', ["seed=", "preprocess"])
	except getopt.GetoptError as err:
		# print help information and exit:
		print(err) # will print something like "option -a not recognized"
		usage()
		sys.exit(2)
	for o, a in optlist:
		if o in ("-s","--seed"):
			seed = a
		elif o in("-p","--preprocess"):
			prep = True
		else:
			assert False, "unhandled option"

	if prep:
		if len(argv) < 1:
			print('Please enter the directory to load authors from.')
			sys.exit(2)
		else:
			p.organize_dataset(seed, argv[0])
			p.organize_authors()
	else:
		if len(argv) < 2:
			print('Please enter training and test directories.')
			sys.exit(2)
		else:
			p.organize_authors(argv[0], argv[1])

	test_authors(p)
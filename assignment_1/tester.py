#!/usr/bin/env python3
from preprocessor import Preprocessor
from tokenizer import Tokenizer
from naive_bayes import MultinomialNaiveBayes, BinarizedMultinomialNaiveBayes, NormalizingNaiveBayes
import numpy as np
import getopt
import sys

def f_score(precision, recall, beta = 1):
	""" Returns the f-score (harmonic mean) of the given parameters. """
	return ((beta ** 2 + 1) * precision * recall) / ((beta ** 2) * precision + recall)

def print_scores(scores):
		print('Micro-averaged scores:')
		print('Precision:',scores[0])
		print('Recall:', scores[1])
		print('F-score (beta=1):', scores[2])
		print('Macro-averaged scores:')
		print('Precision:',scores[3])
		print('Recall:', scores[4])
		print('F-score (beta=1):', scores[5])

def print_multiple_scores(scores):
	names = ['Bag of Words', 'Bag of Character N-Grams', 'Set of Words', 'Complexity Features']
	for i in range(len(scores)):
		if scores[i] is not None:
			print('Scores for the',names[i],'feature set:')
			print_scores(scores[i])
			print('\n')

def test_authors(p, bag_of_words = True, alpha = 0.05, bag_of_char_ngrams = False, ngram_len = 5, set_of_words = False, complexity_features = False,
	print_predictions = True):
	""" Tests the classifiers with the given feature sets.

	p is the Preprocessor object holding the path data. It must have been initialized by
	calling p.organize_authors()

	If bag_of_words argument is True, the program uses Multinomial Naive Bayes classifier
	with the given alpha and bag of words as the feature set.

	If bag_of_char_ngrams argument is True, the program uses Multinomial Naive Bayes classifier
	with the given alpha, and a bag of character n-grams as the feature set.

	If set_of_words argument is True, the program uses Binarized Multinomial Naive Bayes clasifier
	with the given alpha, and a set of words as the feature set.

	If complexity_features argument is True, the program uses Normalizing Naive Bayes, which
	is my term for a classifier which simply fits all features for all classes into their own normal
	distributions and calculates probabilities using the pdfs.

	Returns a 4-tuple, each being the score tuple a different feature set, in the order they are written
	above. Any feature sets not used will return a score of None.
	"""
	authors = p.get_authors()
	classifiers = (None if not bag_of_words else MultinomialNaiveBayes(authors, alpha = alpha),
		None if not bag_of_char_ngrams else MultinomialNaiveBayes(authors, alpha = alpha),
		None if not set_of_words else BinarizedMultinomialNaiveBayes(authors, alpha = alpha),
		None if not complexity_features else NormalizingNaiveBayes(authors, 8))

	# Train the bayes classifiers for each training data
	for author in authors:
		for clsf in classifiers:
			if clsf is not None: clsf.add_documents(author, len(p.training_data(author)))

		for data in p.training_data(author):

			# Featurize and add features to the classifiers
			t = Tokenizer(p.file_path(author,data))
			if classifiers[0] is not None: classifiers[0].add_feature_counts(author, t.bag_of_words())
			if classifiers[1] is not None: classifiers[1].add_feature_counts(author, t.bag_of_char_ngrams(ngram_len))
			if classifiers[2] is not None: classifiers[2].add_feature_counts(author, t.bag_of_words())
			if classifiers[3] is not None: classifiers[3].add_features(author, classifiers[3].vectorize(t.features()))

	for clsf in classifiers:
		if clsf is not None: clsf.train()

	testers = (None if not bag_of_words else Tester(classifiers[0].get_classes()),
		None if not bag_of_char_ngrams else Tester(classifiers[1].get_classes()),
		None if not set_of_words else Tester(classifiers[2].get_classes()),
		None if not complexity_features else Tester(classifiers[3].get_classes()))

	# Check the classifier predictions for each test data
	for author in authors:
		for data in p.test_data(author):

			# Featurize and classify
			t = Tokenizer(p.file_path(author,data, training_data = False))
			class_predicted = [None, None, None, None]
			if classifiers[0] is not None:
				class_predicted[0] = classifiers[0].most_probable_class(classifiers[0].vectorize(t.bag_of_words()))
				testers[0].add_stat(class_predicted[0], author)
			if classifiers[1] is not None:
				class_predicted[1] = classifiers[1].most_probable_class(classifiers[1].vectorize(t.bag_of_char_ngrams(ngram_len)))
				testers[1].add_stat(class_predicted[1], author)
			if classifiers[2] is not None:
				class_predicted[2] = classifiers[2].most_probable_class(classifiers[2].vectorize(t.bag_of_words()))
				testers[2].add_stat(class_predicted[2], author)
			if classifiers[3] is not None:
				class_predicted[3] = classifiers[3].most_probable_class(classifiers[3].vectorize(t.features()))
				testers[3].add_stat(class_predicted[3], author)
			if print_predictions: print('predicted:',[pr for pr in class_predicted if pr is not None],'actual:',author)
		
	return (testers[0].scores() if testers[0] is not None else None, testers[1].scores() if testers[1] is not None else None,
		testers[2].scores() if testers[2] is not None else None, testers[3].scores() if testers[3] is not None else None)

class Tester:
	""" Tester for a single Naive Bayes classifier. """
	def __init__(self, classes):
		cls_len = len(classes)
		self.classes = classes
		self.stats = np.zeros((cls_len, cls_len), dtype=np.int)

	def add_stat(self, predicted_class, real_class):
		""" Adds a single stat to the confusion matrix. """
		self.stats[self.classes[real_class],self.classes[predicted_class]] += 1

	def microavg_precision(self):
		""" Returns the micro-averaged precision value. """
		return np.trace(self.stats) / np.sum(self.stats)

	def microavg_recall(self):
		""" Returns the micro-averaged recall value. """
		return np.trace(self.stats) / np.sum(self.stats)

	def macroavg_precision(self):
		""" Returns the macro-averaged precision value.
		Note that nan values are ignored.
		"""
		with np.errstate(divide='ignore', invalid='ignore'):
			return np.nanmean(np.divide(np.diagonal(self.stats), np.sum(self.stats, axis=0)))

	def macroavg_recall(self):
		""" Returns the macro-averaged recall value. """
		with np.errstate(divide='ignore', invalid='ignore'):
			return np.nanmean(np.divide(np.diagonal(self.stats), np.sum(self.stats, axis=1)))

	def scores(self):
		""" Returns a 6-tuple of scores. """
		micro_prec = self.microavg_precision()
		micro_rec = self.microavg_recall()
		micro_f = f_score(micro_prec, micro_rec)
		macro_prec = self.macroavg_precision()
		macro_rec = self.macroavg_recall()
		macro_f = f_score(macro_prec, macro_rec)
		return micro_prec, micro_rec, micro_f, macro_prec, macro_rec, macro_f

if __name__ == '__main__':
	""" This program normally accepts two arguments: the directory of the training set and
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
	with np.errstate(divide='ignore', invalid='ignore'):	
		scores = test_authors(p, bag_of_words = True, alpha = 0.05, bag_of_char_ngrams = True, ngram_len = 5,
			set_of_words = False, complexity_features = False, print_predictions = False)
		print_multiple_scores(scores)
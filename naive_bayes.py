#!/usr/bin/env python3
from preprocessor import Preprocessor
from tokenizer import Tokenizer
from collections import Counter
from scipy.stats import norm
import numpy as np
import math
import sys

class NaiveBayes:
	""" Base class for Naive Bayes implementations.
	Defines the method of calculating the most probable class.
	Classes must implement the calculate_class_log_probability method
	in a vectorized fashion.
	"""
	def __init__(self,classes):
		self.classes = dict(zip(classes, range(0, len(classes))))
		self.class_indices = dict([reversed(i) for i in self.classes.items()])
		self.documents = np.zeros((len(classes),))
		self.doc_arr = np.zeros((len(classes),))

	def get_classes(self):
		""" Convenience method for getting classes and their indexes. """
		return self.classes

	def add_documents(self, class_name, amount = 1):
		""" Adds a number of documents to the given class.
		This method should be called until the total amount matches the number of documents
		in the training set for each class.
		"""
		self.documents[self.classes[class_name]] += amount

	def train(self):
		""" Handles the calculation of class base probabilities. """
		d = np.array(self.documents)
		d = d / np.sum(d)
		self.doc_arr = d

	def most_probable_class(self, feature_vec):
		""" Returns the class with the highest log probability, given the list of features. """	
		return self.class_indices[np.argmax(self.class_log_probabilities(feature_vec))]

	def class_log_probabilities(self, feature_vec, normalize = False):
		""" Returns the log probabilities of all classes in a vector,
		where each index corresponds to the class given by self.class_indices[index].
		If normalize is True, all log probabilities are normalized to [0,1] interval
		(1 being the highest log probability) while retaining their orderings.
		"""
		probs = self.doc_arr + self.calculate_class_log_probability(feature_vec)
		if not normalize: return probs
		pos_probs = probs-np.min(probs)
		return pos_probs / np.max(pos_probs)

	def calculate_class_log_probability(self, feature_vec):
		""" Gets the log probabilities of the feature vector with respect to classes. """
		pass

class MultinomialNaiveBayes(NaiveBayes):
	""" Implementation of Multinomial Naive Bayes classifier. """
	def __init__(self, classes, alpha = 1):
		""" Constructs a multinomial naive bayes classifier.
		Alpha is the Laplace smoothing parameter.
		"""
		NaiveBayes.__init__(self, classes)
		self.class_data = {}
		for class_name in classes:
			self.class_data[class_name] = Counter()
		self.alpha = alpha
		self.class_features = [[]]
		self.features = {}

	def train(self):
		""" Trains the classifier after all features are added. """
		NaiveBayes.train(self)
		features = Counter()
		for c in self.class_data.values():
			features.update(c)
		self.features = dict(zip(features.keys(), range(0, len(features))))
		self.class_features = np.zeros((len(self.classes),len(self.features)+1))
		for (class_name,class_index) in self.classes.items():
			v = self.vectorize(self.class_data[class_name])
			self.class_features[class_index,:] = self.log_prob(v)

	def log_prob(self, v):
		""" Calculates the log probability of a feature vector.
		Uses Laplace smoothing with alpha parameter. """
		return np.log(np.divide(v + self.alpha, np.sum(v) + self.alpha * len(self.features)))

	def vectorize(self, features):
		""" Vectorizes the given feature set according to the trained feature set.
		Any features that are not found in the training set are put into the last index
		of the feature vector.

		Parameter features is a dictionary of features to feature counts.
		"""
		v = [features[feature] for feature in self.features.keys()]
		v.append(sum([0 if feature in self.features else fcount for (feature,fcount) in features.items()]))
		return np.array(v)

	def add_feature_counts(self, class_name, counts):
		""" Adds the given features to a class.
		Parameter counts is a dictionary of features to feature counts.
		"""
		self.class_data[class_name] += counts

	def calculate_class_log_probability(self, feature_vec):
		""" Gets the log probability of a feature vector being in each class. """
		return (self.class_features.dot(feature_vec.T)).T

class BinarizedMultinomialNaiveBayes(MultinomialNaiveBayes):
	""" Binarized version of the Multinomial Naive Bayes classifier. """
	def vectorize(self, features):
		""" Vectorizes the given feature set according to the trained feature set.
		Any features that are not found in the training set are put into the last index
		of the feature vector.

		Parameter features is a dictionary of features to feature counts. Since this
		implementation is binarized, the counts are set to 1 if they are non-zero.
		"""
		v = MultinomialNaiveBayes.vectorize(self,features)
		return np.minimum(np.ones(len(v)),v)

class NormalizingNaiveBayes(NaiveBayes):
	""" A Naive Bayes implementation that tries to fit features into normal distributions.
	The probability of a feature given a class then becomes the pdf of that feature for each
	occurance.
	"""
	def __init__(self, classes, num_features):
		""" Initializes the necessary components. """
		NaiveBayes.__init__(self, classes)
		self.class_data = {}
		for class_name in classes:
			self.class_data[class_name] = [[] for i in range(num_features)]
		self.feature_means = None
		self.feature_stddevs = None
		self.num_features = num_features

	def add_features(self, class_name, features):
		""" Adds a realization of a feature set to a class. """
		for i in range(self.num_features):
			self.class_data[class_name][i].append(features[i])

	def fit(self, features):
		""" Fits each of the given tuple of features to a normal distribution. """
		return [norm.fit(feature) for feature in features]

	def vectorize(self, features):
		""" Vectorizes the tuple of features. """
		return np.array([features[0],np.mean(features[1]),np.mean(features[2]),np.mean(features[3]),np.mean(features[4]),np.mean(features[5]),np.mean(features[6]),features[7]])

	def train(self):
		""" Trains the naive bayes classifier. """
		NaiveBayes.train(self)
		features = np.array([(lambda x: self.fit(self.class_data[x]))(class_name) for class_name in self.classes.keys()])
		self.feature_means = np.matrix(features[:,:,0])
		self.feature_stddevs = np.matrix(features[:,:,1])

	def calculate_class_log_probability(self, feature_vec):
		""" Gets the log probability of the feature vector for each class """
		p = norm.pdf(feature_vec, self.feature_means, self.feature_stddevs)
		return np.nansum(np.log(p),axis = 1)

if __name__ == '__main__':
	""" Performs the basic example from the lecture notes. """
	bayes = MultinomialNaiveBayes(['china','japan'])
	china = 'Chinese Beijing Chinese Chinese Chinese Shanghai Chinese Macao'.split(' ')
	japan = 'Tokyo Japan Chinese'.split(' ')
	bayes.add_documents('china', amount = 3)
	bayes.add_documents('japan', amount = 1)
	bayes.add_feature_counts('china', Counter(china))
	bayes.add_feature_counts('japan', Counter(japan))
	bayes.train()
	print(bayes.most_probable_class(bayes.vectorize('Chinese Chinese Chinese Tokyo Japan'.split(' '))))
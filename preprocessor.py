#!/usr/bin/env python3
import math
import os
import random
import sys
import shutil

class Preprocessor:
	def __init__(self):
		self.training_path = ''
		self.test_path = ''
		self.authors = {}

	def organize_dataset(self, seed, path, ratio = 0.6):
		""" Generates training and test datasets for each author.
		The training and test datasets are copied to the new directories.
		The texts belonging to authors are shuffled with Fisher-Yates shuffle
		in order to make the test set selection random.

		Seed is the seed value given to the psuedo-random number generator
		so that the same training set can be generated if needed. Path is the
		outermost directory that houses the author directories. Ratio defines the ratio of training
		dataset to the whole dataset.

		Returns the training and test paths.
		"""
		if seed is not None:
			random.seed(seed)
		path = os.path.normpath(path)
		self.training_path = path + '__training'
		self.test_path = path + '__test'
		self.init_dirs(self.training_path, self.test_path)
		authors = os.listdir(path)
		for author in authors:
			texts = os.listdir(os.path.join(path, author))
			os.makedirs(os.path.join(self.training_path, author), exist_ok=True)
			os.makedirs(os.path.join(self.test_path, author), exist_ok=True)
			random.shuffle(texts)
			train_size = math.floor(ratio*len(texts))
			for text in texts[0:train_size]:
				shutil.copyfile(os.path.join(path, author, text), os.path.join(self.training_path, author, text))
			for text in texts[train_size:]:
				shutil.copyfile(os.path.join(path, author, text), os.path.join(self.test_path, author, text))
		return self.training_path, self.test_path

	def init_dirs(self, training_path, test_path):
		""" Initializes the training and test directories from scratch. """
		shutil.rmtree(training_path, ignore_errors=True)
		shutil.rmtree(test_path, ignore_errors=True)
		os.makedirs(training_path, exist_ok=True)
		os.makedirs(test_path, exist_ok=True)

	def organize_authors(self, training_path = None, test_path = None):
		""" Generates the internal representation of training and test sets for each author.
		If training and test paths are set (ex: as a result of calling organize_dataset()),
		their arguments are not necessary. Authors that do not have any training data
		are automatically ignored.
		"""
		if training_path is not None and test_path is not None:
			self.training_path = training_path
			self.test_path = test_path
		training_authors = os.listdir(self.training_path)
		for author in training_authors:
			self.authors[author] = { 'training':[], 'test':[] }
			self.authors[author]['training'] = os.listdir(os.path.join(self.training_path, author))
			self.authors[author]['test'] = os.listdir(os.path.join(self.test_path, author))

	def get_authors(self):
		""" Convenience method for getting the list of authors.

		Returns the list of authors.
		"""
		return self.authors.keys()

	def training_data(self, author):
		""" Convenience method for getting the training set of a certain author.

		Returns the list of file names selected as training data.
		"""
		return self.authors[author]['training']

	def test_data(self, author):
		""" Convenience method for getting the test set of a certain author.

		Returns the list of file names selected as tes data.
		"""
		return self.authors[author]['test']

	def file_path(self, author, file_name, training_data = True):
		""" Returns the path to the file with the given parameters. Does not do any checks. """
		if training_data:
			return os.path.join(self.training_path, author, file_name)
		else:
			return os.path.join(self.test_path, author, file_name)

if __name__ == '__main__':
	""" Accepts two arguments, one mandatory (authors directory) and one optional
	(seed for random number generation)
	"""
	if len(sys.argv) < 2:
		print('Please enter the directory to load authors from.')
	else:
		seed = None
		if len(sys.argv) > 2:
			seed = sys.argv[2]

		p = Preprocessor()
		p.organize_dataset(seed, sys.argv[1])
		p.organize_authors()
#!/usr/bin/env python3
import math
import os
import random
import sys
import shutil

class Preprocessor:
	def __init__(self, seed, path, ratio = 0.6, auto_organize = False):
		"""
		Initializes the object. Seed is the seed value given to the psuedo-random number generator
		so that the same training set can be generated if needed. Path is the
		outermost directory that houses the author directories. Ratio defines the ratio of training
		dataset to the whole dataset. If auto_organize is set to True, the initializer will
		automatically generate the data without the need to call organize() method externally
		"""
		random.seed(seed)
		self.path = os.path.split(path)[0]
		self.ratio = ratio
		self.authors = {}
		if auto_organize:
			self.organize()

	def init_dirs(self):
		"""
		Initializes the training and test directories from scratch.
		"""
		shutil.rmtree(self.path+'__training', ignore_errors=True)
		shutil.rmtree(self.path+'__test', ignore_errors=True)
		os.makedirs(self.path+'__training', exist_ok=True)
		os.makedirs(self.path+'__test', exist_ok=True)

	def organize(self):
		"""
		Generates training and test datasets for each author and copies the related
		files to the new directories. The texts belonging to authors are shuffled
		with Fisher-Yates shuffle in order to make the test set selection random.
		"""
		self.init_dirs()
		authors = os.listdir(self.path)
		for author in authors:
			self.authors[author] = { 'training':[], 'test':[] }
			texts = os.listdir(os.path.join(self.path, author))
			os.makedirs(os.path.join(self.path+'__training', author), exist_ok=True)
			os.makedirs(os.path.join(self.path+'__test', author), exist_ok=True)
			random.shuffle(texts)
			trainSize = math.floor(self.ratio*len(texts))
			for text in texts[0:trainSize]:
				self.authors[author]['training'].append(text)
				shutil.copyfile(os.path.join(self.path, author, text), os.path.join(self.path+'__training', author, text))
			for text in texts[trainSize:]:
				self.authors[author]['test'].append(text)
				shutil.copyfile(os.path.join(self.path, author, text), os.path.join(self.path+'__test', author, text))

	def get_authors(self):
		"""
		Convenience method for getting the list of authors.

		Returns the list of authors.
		"""
		return self.authors.keys()

	def training_data(self, author):
		"""
		Convenience method for getting the training set of a certain author.

		Returns the list of file names selected as training data.
		"""
		return self.authors[author]['training']

	def test_data(self, author):
		"""
		Convenience method for getting the test set of a certain author.

		Returns the list of file names selected as tes data.
		"""
		return self.authors[author]['test']

	def file_path(self, author, file_name, training_data = True):
		"""
		Returns the path to the file with the given parameters. Does not do any checks.
		"""
		if training_data:
			return os.path.join(self.path+'__training', author, file_name)
		else:
			return os.path.join(self.path+'__test', author, file_name)

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
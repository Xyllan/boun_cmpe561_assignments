#!/usr/bin/env python3
from collections import OrderedDict
import sys
import train_hmm_tagger as hmm_train
import conll_parser as cpar
import numpy as np

def get_pred_sentences(path):
	""" Gets the required portion of the sentences from a path.
	Each element of the sentence is a 4-tuple, denoting the
	form, lemma, cpostag, and postag.

	Works on inputs like the example test output, ie. split by |.
	"""
	lines = []
	with open(path, 'r', encoding = 'utf-8') as fp:
		lines = fp.readlines()
	sentences = []
	acc = []
	for line in lines:
		word = line.strip().split('|')
		if not len(word) < 2:
			acc.append((word[0],word[1],word[1]))
		else: # End of a sentence.
			if acc is not []:
				sentences.append(acc)
				acc = []
	return sentences 

class Tester:
	""" Tester for HMM POS tagger. """
	def __init__(self, tags):
		tag_len = len(tags)
		self.tags = OrderedDict(zip(tags, range(0, len(tags))))
		self.stats = np.zeros((tag_len, tag_len), dtype=np.int)

	def build(self, gold_sentences, predicted_sentences):
		""" Builds the confusion matrix.
		The given predicted sentences and the gold standard sentences
		must be of the same length. The total number of sentences must
		also match.
		""" 
		for i, gold_sent in enumerate(sentences):
			pr_sent = predicted_sentences[i]
			for j, gold_word in enumerate(gold_sent):
				pr_word = pr_sent[j]
				t.add_stat(pr_word[hmm.tag_ind],gold_word[hmm.tag_ind])
	
	def add_stat(self, predicted_tag, real_tag):
		""" Adds a single stat to the confusion matrix. """
		self.stats[self.tags[real_tag],self.tags[predicted_tag]] += 1

	def accuracy(self, tag):
		""" Calculates the accuracy of a single tag. """
		ind = self.tags[tag]
		s = np.sum(self.stats)
		return (s - np.sum(self.stats[ind,:]) - np.sum(self.stats[:,ind]) + 2 * self.stats[ind,ind]) / s

	def overall_accuracy(self):
		""" Calculates the overall accuracy. """
		return np.trace(self.stats) / np.sum(self.stats)

	def print_acc(self):
		""" Prints the accuracies of all tags, plus the overall accuracy. """
		print('Accuracies:')
		print('Overall Accuracy:',self.overall_accuracy())
		for tag in self.tags.keys():
			print(tag,'Accuracy:',self.accuracy(tag))

	def print_conf(self):
		""" Prints the confusion matrix. """
		print('Confusion matrix:')
		print('  Tags:',list(self.tags.keys()))
		print(self.stats)

if __name__ == '__main__':
	""" This program accepts two arguments: the file path to generated output
	file and the file path to the gold standard file. It requires a hmm.conf file to 
	have been created by train_hmm_tagger.py
	"""
	argv = sys.argv[1:]
	if len(argv) < 2:
		print('You must enter a output filepath and gold standard filepath.')
		sys.exit(2)
	else:
		output_filepath = argv[0]
		gold_filepath = argv[1]

		hmm = hmm_train.HMM()
		hmm.load(hmm_train.config_path)

		sentences = cpar.get_sentences(gold_filepath)
		pr_sentences = get_pred_sentences(output_filepath)

		t = Tester(hmm.tags)
		t.build(sentences,pr_sentences)
		
		t.print_acc()
		t.print_conf()
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
		self.unk_stats = np.zeros((tag_len, tag_len), dtype=np.int)
		self.knw_stats = np.zeros((tag_len, tag_len), dtype=np.int)

	def build(self, gold_sentences, predicted_sentences, vocab = set([])):
		""" Builds the confusion matrix.
		The given predicted sentences and the gold standard sentences
		must be of the same length. The total number of sentences must
		also match.
		""" 
		for i, gold_sent in enumerate(sentences):
			pr_sent = predicted_sentences[i]
			for j, gold_word in enumerate(gold_sent):
				pr_word = pr_sent[j]
				known = pr_word[0] in vocab
				self.add_stat(pr_word[hmm.tag_ind],gold_word[hmm.tag_ind], known = known)
	
	def get_stats(self, stat_type = 2):
		""" Gets the stat matrix of the relevant type.
		Stat Type 0: Unknown words only
		Stat Type 1: Known words only
		Stat Type 2: All words
		"""
		if stat_type is 0:
			return self.unk_stats
		elif stat_type is 1:
			return self.knw_stats
		else:
			return self.unk_stats + self.knw_stats

	def add_stat(self, predicted_tag, real_tag, known = True):
		""" Adds a single stat to the confusion matrix. """
		if known:
			self.knw_stats[self.tags[real_tag],self.tags[predicted_tag]] += 1
		else:
			self.unk_stats[self.tags[real_tag],self.tags[predicted_tag]] += 1

	def accuracy(self, tag, stat_type = 2):
		""" Calculates the accuracy of a single tag.
		See get_stats function for information on stat_type 
		"""
		ind = self.tags[tag]
		stats = self.get_stats(stat_type)
		s = np.sum(stats)
		return (s - np.sum(stats[ind,:]) - np.sum(stats[:,ind]) + 2 * stats[ind,ind]) / s

	def overall_accuracy(self, stat_type = 2):
		""" Calculates the overall accuracy.
		See get_stats function for information on stat_type 
		"""
		stats = self.get_stats(stat_type)
		return np.trace(stats) / np.sum(stats)

	def print_acc(self, stat_type = 2):
		""" Prints the accuracies of all tags, plus the overall accuracy. """
		print('Accuracies:')
		print('Overall Accuracy:',self.overall_accuracy(stat_type))
		for tag in self.tags.keys():
			print(tag,'Accuracy:',self.accuracy(tag, stat_type))

	def print_conf(self, stat_type = 2):
		""" Prints the confusion matrix. """
		print('Confusion matrix:')
		print('  Tags:',list(self.tags.keys()))
		print(self.get_stats(stat_type))

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
		t.build(sentences,pr_sentences, vocab = hmm.vocab)
		
		print('Stats for unknown words:')
		t.print_acc(0)
		t.print_conf(0)
		print('Stats for known words:')
		t.print_acc(1)
		t.print_conf(1)
		print('Stats for all words:')
		t.print_acc(2)
		t.print_conf(2)
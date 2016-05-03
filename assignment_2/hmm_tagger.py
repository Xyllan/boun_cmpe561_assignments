#!/usr/bin/env python3
import io
import sys
import train_hmm_tagger as hmm_train
import conll_parser as cpar

def pos_tag(hmm, sentences):
	""" Returns POS tagged versions of the given sentences. """
	return [viterbi(hmm,sentence) for sentence in sentences]

def find_best_parent(hmm, word, tag, parents):
	""" Finds the best parent for the given word/tag tuple.
	Part of the viterbi algorithm.
	"""
	# Try the current word with a tag.
	trial = (word, tag, tag)
	max_lp = float('-inf')
	max_lp_parent_ind = -1

	# Find the parent that would maximize the log probability
	# with the current tag.
	for k, (prev_word_tpl, parent, log_prob) in enumerate(parents):
		lp = hmm.word_log_prob(prev_word_tpl[hmm.tag_ind], tag, trial[0]) + log_prob
		if lp >= max_lp:
			max_lp = lp
			max_lp_parent_ind = k

	return (trial, max_lp_parent_ind, max_lp)


def viterbi(hmm, sentence):
	""" Implements the Viterbi algorithm for HMMs. """
	# Find the most common tag
	max_lp_tag = None
	max_lp = float('-inf')
	for tag in hmm.tags:
		lp = hmm.tag_count(tag)
		if lp >= max_lp:
			max_lp = lp
			max_lp_tag = tag
	
	v = []
	# The initial node for all paths is the start tag
	start = (None, hmm_train.start_tag, hmm_train.start_tag)
	v.insert(0,[(start, -1, 0)])

	for i, word_tpl in enumerate(sentence, start=1): # Columns
		v.insert(i,[])
		if word_tpl[0] in hmm.vocab: # Word is in our vocabulary
			for j, tag in enumerate(hmm.tags): # Rows
				bp = find_best_parent(hmm, word_tpl[0],tag, v[i-1])
				# Add the current word/tag pair, its parent, and its log probability to the array.
				v[i].insert(j,bp)
		else: # Word is not in our vocabulary, assign the most common tag
			v[i].insert(0, find_best_parent(hmm, word_tpl[0], max_lp_tag, v[i-1]))
	
	# Get the word/tag pair with the highest log probability.
	max_lp_word = None
	max_lp = float('-inf')
	for word in v[-1]:
		lp = hmm.end_log_prob(word[0]) + word[2] # Make sure to calculate the end probability
		if lp >= max_lp:
			max_lp = lp
			max_lp_word = word

	ind = len(v)-2
	sent = []
	# Iterate over the parents until the beginning.
	while(max_lp_word[1] is not -1):
		sent.insert(0, max_lp_word[0])
		max_lp_word = v[ind][max_lp_word[1]]
		ind -= 1
	return sent

def save(ind, sentences, output_filepath):
	""" Outputs the tagged sentences to the given filepath. """
	with io.open(output_filepath, 'w', encoding='utf-8') as f:
		for sentence in sentences:
			for word_tpl in sentence:
				f.write(word_tpl[0]+'|'+word_tpl[ind]+'\n')
			f.write('\n')
	print('Output written to',output_filepath)
	


if __name__ == '__main__':
	""" This program accepts two arguments: the file path to the test file
	and the file path to the output file. It requires a hmm.conf file to 
	have been created by train_hmm_tagger.py
	"""
	argv = sys.argv[1:]
	if len(argv) < 2:
		print('You must enter a test filepath and output filepath')
		sys.exit(2)
	else:
		test_filepath = argv[0]
		output_filepath = argv[1]

		hmm = hmm_train.HMM()
		hmm.load(hmm_train.config_path)

		sentences = cpar.get_sentences(test_filepath)

		pt_sentences = pos_tag(hmm, sentences)

		save(hmm.tag_ind, pt_sentences, output_filepath)
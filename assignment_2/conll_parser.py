#!/usr/bin/env python3
import sys 

def get_sentences(path):
	""" Gets the required portion of the sentences from a path.
	Each element of the sentence is a 4-tuple, denoting the
	form, lemma, cpostag, and postag.
	"""
	lines = []
	with open(path, 'r', encoding = 'utf-8') as fp:
		lines = fp.readlines()
	sentences = []
	acc = []
	for line in lines:
		word = line.strip().split('\t')
		if not len(word) < 2:
			if word[1] is not '_':
				try:
					acc.append((word[1],word[3],word[4]))
				except IndexError: # There is no postag or cpostag.
					acc.append((word[1],None,None))
		else: # End of a sentence.
			sentences.append(acc)
			acc = []
	sentences.append(acc)
	return sentences 

def tag_ind(tag_type = 'cpostag'):
	return 1 if tag_type is 'cpostag' else 2

def tag_list(sentences, tag_ind = 1):
	""" Gets the total list of tags. """
	return set([word[tag_ind] for sentence in sentences for word in sentence])

if __name__ == '__main__':
	""" Basic program to get a list of cpostags. """
	if len(sys.argv) < 2:
		print('You must enter a file path')
	else:
		path = sys.argv[1]
		sentences = get_sentences(path)
		print(tag_list(sentences))
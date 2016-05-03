#!/usr/bin/env python3
import argparse
import conll_parser as cpar
import io, json
import math

start_tag = '<s>'
end_tag = '<e>'
config_path = 'hmm.conf'

def is_sequence(arg):
	""" Helper for figuring if an object is a sequence.
	Lists and tuples are considered a sequence.
	Strings are NOT considered a sequence.
	"""
	return (not hasattr(arg, "strip") and (hasattr(arg, "__getitem__") or hasattr(arg, "__iter__")))

def to_dict(obj, count):
	""" Constructs a dictionary object from the given object.
	If the given object refers to a tag/word pair, it will be of the form:
	(str,str)
	If the given object refers to a tag/prev-tag pair, it will be of the form:
	(str,str)
	If the given object refers to a tag, it will be of the form:
	str
	"""
	if is_sequence(obj): # Count of a tag/word or tag/prev-tag pair
		return {'tag':obj[1], 'str':obj[0], 'count':count}
	else: # Count of a tag
		return {'tag':obj, 'count':count}

def from_dict(dict_obj):
	""" Reverses the to_dict function above. """
	obj = None
	if 'str' in dict_obj: # Count of a tag/word or tag/prev-tag pair
		obj = (dict_obj['str'],dict_obj['tag'])
	else: # Count of a tag
		obj = dict_obj['tag']

	return obj, dict_obj['count']

class HMM:
	""" A Hidden Markov Model implementation.
	Constructed for bi-gram POS tagging.
	"""
	def __init__(self, tags = set([]), tag_ind = -1):
		self.tags = tags
		self.tag_ind = tag_ind
		self.counts = {}
		self.vocab = set([])

	def add_count(self, obj, amount = 1):
		""" Adds an amount to the given count object. """
		if obj in self.counts:
			self.counts[obj]+=amount
		else:
			self.counts[obj]=amount

	def add_tag(self, tag, amount = 1):
		""" Adds an amount of tag occurences.
		The count object for a tag is:
		'tag_name'
		"""
		self.add_count(tag, amount)

	def tag_count(self, tag):
		""" Convenience method for getting a tag count. """
		try:
			return self.counts[tag]
		except KeyError:
			return 0

	def add_tag_pair(self, prev_tag, tag, amount = 1):
		""" Adds an amount of previous tag, tag occurences.
		The count object for a previous tag, tag pair is:
		('prev_tag_name', 'tag_name')
		"""
		self.add_count((prev_tag, tag), amount)

	def tag_pair_count(self, prev_tag, tag):
		""" Convenience method for getting a tag pair count. """
		try:
			return self.counts[(prev_tag,tag)]
		except KeyError:
			return 0

	def add_word_tag_pair(self, word, tag, amount = 1):
		""" Adds an amount of word tag occurences.
		The count object for a word tag pair is:
		('word','tag_name')
		"""
		self.add_count((word,tag), amount)
		self.vocab.add(word)

	def word_tag_count(self, word, tag):
		""" Convenience method for getting a tag word count. """
		try:
			return self.counts[(word,tag)]
		except KeyError:
			return 0

	def add_word_tuple(self, word, prev_word, amount = 1):
		""" Adds an amount of word bi-gram occurences. """
		self.add_tag(word[self.tag_ind], amount)
		self.add_tag_pair(prev_word[self.tag_ind],word[self.tag_ind], amount)
		self.add_word_tag_pair(word[0],word[self.tag_ind], amount)

	def train(self, sentences):
		""" Trains the HMM with the given sentences. """
		for sentence in sentences:
			prev_word = (None, start_tag, start_tag)
			# Count the start states.
			self.add_tag(start_tag)
			self.add_word_tag_pair(None,start_tag)
			for word in sentence:
				self.add_word_tuple(word,prev_word)
				prev_word = word
			# Count the end state.
			self.add_tag_pair(prev_word[self.tag_ind],end_tag)

	def save(self, path = config_path):
		""" Saves the current state of the HMM to a file. """
		with io.open(path, 'w', encoding='utf-8') as f:
			data = {'tags':list(self.tags),
					'vocab':list(self.vocab),
					'tag_ind':self.tag_ind,
					'counts':[to_dict(obj,count) for obj,count in self.counts.items()]
					}
			f.write(json.dumps(data, ensure_ascii=False))
		print('HMM configuration saved to',path)

	def load(self, path = config_path):
		""" Loads the current state of the HMM from a file. """
		with io.open(path, 'r', encoding = 'utf-8') as f:
			data = json.load(f)
			self.tags = set(data['tags'])
			self.vocab = set(data['vocab'])
			self.tag_ind = data['tag_ind']
			self.counts = dict([from_dict(c) for c in data['counts']])
		print('HMM configuration loaded from',path)

	def word_log_prob(self, prev_tag, tag, word):
		""" Gets the log probability of a word being an instance of the
		given tag, given the previous tag.

		Since we are using a bi-gram model, this probability is equal to:
		Count(prev_tag,tag)/Count(prev_tag) * Count(word, tag)/Count(tag)

		Note that if the log probability evaluates to negative infinity,
		we use -1e10 in order to make meaningful choices down the road,
		which pays off if all posterior probabilities evaluate to 0.
		"""
		try:
			return math.log(self.tag_pair_count(prev_tag, tag)) + math.log(self.word_tag_count(word, tag)) - \
			math.log(self.tag_count(tag)) - math.log(self.tag_count(prev_tag))
		except ValueError:
			return -1e10

	def end_log_prob(self, tag):
		""" Gets the log probability of having a tag at the end of a sentence.
		This probability is equal to:
		Count(prev_tag,end_tag) / Count(prev_tag)
		"""
		try:
			return math.log(self.tag_pair_count(tag, end_tag)) - math.log(self.tag_count(prev_tag))
		except ValueError:
			return -1e10

if __name__ == '__main__':
	""" This program accepts one argument: the file path to the training set.
	If used, the -c or --cpostag option will make the program use the cpostags.
	The -p or --postag option will make the program use the postags.
	If both options are omitted, the program will use the cpostag.

	When done, this program will save its HMM configuration and exit.
	"""
	tag_type = 'cpostag'

	parser = argparse.ArgumentParser()
	parser.add_argument("training_filepath", help="path to training file")
	parser.add_argument("-c", "--cpostag", help="uses cpostags", action="store_true")
	parser.add_argument("-p", "--postag", help="uses postags", action="store_true")
	args = parser.parse_args()
	if args.cpostag:
		tag_type = 'cpostag'
	elif args.postag:
		tag_type = 'postag'
	else:
		print('Using cpostags since tag set was not specified.')

	sentences = cpar.get_sentences(args.training_filepath)
	tag_ind = cpar.tag_ind(tag_type)
	hmm = HMM(cpar.tag_list(sentences, tag_ind), tag_ind)
	hmm.train(sentences)

	hmm.save()
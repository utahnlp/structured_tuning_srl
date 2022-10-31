import sys
import argparse
import numpy as np
from .preprocess import Indexer, load_frameset
import spacy
from spacy.tokenizer import Tokenizer
import h5py

spacy_nlp = spacy.load('en_core_web_sm')
spacy_nlp.tokenizer = Tokenizer(spacy_nlp.vocab)	# supposedly it tells spacy to only split on space


#def get_possible_arg_names(label_dict, args):
#	# is conll05 mode or conll2012
#	# all modifiers will be enabled no matter what
#	if 'B-A0' in label_dict:
#		return ['B-A{}'.format(a) for a in args] + ['I-A{}'.format(a) for a in args] + ['B-C-A{}'.format(a) for a in args] + \
#			['I-C-A{}'.format(a) for a in args] + ['B-R-A{}'.format(a) for a in args] + ['I-R-A{}'.format(a) for a in args] + \
#			[n for n in label_dict.keys() if 'AM' in n]
#	elif 'B-ARG0' in label_dict:
#		return ['B-ARG{}'.format(a) for a in args] + ['I-ARG{}'.format(a) for a in args] + ['B-C-ARG{}'.format(a) for a in args] + \
#			['I-C-ARG{}'.format(a) for a in args] + ['B-R-ARG{}'.format(a) for a in args] + ['I-R-ARG{}'.format(a) for a in args] + \
#			[n for n in label_dict.keys() if 'ARGM' in n]
#	else:
#		raise Exception('unrecognized label type.')

def get_arg_mask(label_dict, args):
	mask = np.zeros((len(label_dict),))
	for arg in args:
		for l, idx in label_dict.items():
			if arg in l:
				mask[idx] = 1.0

	# always enable modifier and V
	for l, idx in label_dict.items():
		if 'M-' in l or '-V' in l:
			mask[idx] = 1.0

	# always enable *A role
	for l, idx in label_dict.items():
		if 'ARGA' in l or 'AA' in l:
			mask[idx] = 1.0
	return mask

def lemmatize(orig_toks):
	rs = []
	cnt = 0
	for toks in orig_toks:
		act_toks = toks[1:-1]
		lemma = [t.lemma_.lower() for t in spacy_nlp(' '.join(act_toks))]
		lemma = [toks[0]] + lemma + [toks[-1]]
		assert(len(lemma) == len(toks))
		rs.append(lemma)
		cnt += 1
	return rs

# return frame indexer (by lemma)
#	and frame_pool (i.e. vectorized frameset)
def get_frame_pool(roleset_dict, label_dict, frameset):
	frame_indexer = Indexer(symbols=["#"], num_oov=0)	# # stays at the 0-th position
	for lemma, _ in frameset.items():
		frame_indexer.register_all_words([lemma], count=False)

	frame_pool = np.zeros((len(frame_indexer.d), len(roleset_dict), len(label_dict)))
	for i, (lemma, frames) in enumerate(frameset.items()):
		# by default, the first frame is for those mismatches
		if i == 0:
			frame_pool[i, :, :] = 1.0
			continue

		# setup role sets one by one
		for frame_id, args in frames:
			frame_idx = roleset_dict[frame_id]
			frame_pool[i, frame_idx] = get_arg_mask(label_dict, args)

		# enable O for any frame_id, even if that one is not applicable
		frame_pool[i, :, 0] = 1.0


	print('frame_pool shape:', frame_pool.shape)

	return frame_indexer, frame_pool


def load_dict(path):
	rs = {}
	with open(path, 'r') as f:
		for line in f:
			if line.strip() == '':
				continue
			parts = line.split()
			rs[parts[0]] = int(parts[1])
	return rs


def load(path):
	all_lines = []
	with open(path, 'r') as f:
		for l in f:
			if l.rstrip() == '':
				continue
			all_lines.append(l.strip().split())
	return all_lines



def convert(opt, frame_indexer, frame_pool, lemmas, output):
	num_ex = len(lemmas)
	max_seq_l = max([len(p) for p in lemmas])
	frame_idx = np.zeros((num_ex, max_seq_l), dtype=int)

	for ex_idx, sent in enumerate(lemmas):
		for k, lemma in enumerate(sent):
			if lemma in frame_indexer.d:
				frame_idx[ex_idx, k] = frame_indexer.d[lemma]

		if (ex_idx+1) % 10000 == 0:
			print("{}/{} sentences processed".format(ex_idx+1, num_ex))

	# Write output
	f = h5py.File(output, "w")
	f["frame"] = frame_idx
	print("saved {} examples ".format(num_ex))
	f.close()


def process(opt):
	frameset = load_frameset(opt.frameset)
	roleset_dict = load_dict(opt.roleset_dict)
	label_dict = load_dict(opt.label_dict)
	frame_indexer, frame_pool = get_frame_pool(roleset_dict, label_dict, frameset)

	print('saving frame_pool...')
	f = h5py.File(opt.output + '.frame_pool.hdf5', "w")
	f["frame_pool"] = frame_pool
	f.close()

	print('lemmatizing...')
	train_lemma = lemmatize(load(opt.train))
	val_lemma = lemmatize(load(opt.val))
	test1_lemma = lemmatize(load(opt.test1))

	convert(opt, frame_indexer, frame_pool, train_lemma, opt.output + '.train.frame.hdf5')
	convert(opt, frame_indexer, frame_pool, val_lemma, opt.output + '.val.frame.hdf5')
	convert(opt, frame_indexer, frame_pool, test1_lemma, opt.output + '.test1.frame.hdf5')
	if opt.test2 != opt.dir:
		test2_lemma = lemmatize(load(opt.test2))
		convert(opt, frame_indexer, frame_pool, test2_lemma, opt.output + '.test2.frame.hdf5')


def main(arguments):
	parser = argparse.ArgumentParser(
		description=__doc__,
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--roleset_dict', help="Path to frame dict.", default = "conll2012.roleset_id.dict")
	parser.add_argument('--label_dict', help="Path to label dict.",default = "conll2012.label.dict")
	parser.add_argument('--dir', help="Path to the data dir",default = "./data/srl/")
	parser.add_argument('--train', help="Path to training orig_tok_grouped.", default = "conll2012.train.orig_tok_grouped.txt")
	parser.add_argument('--val', help="Path to validation orig_tok_grouped.", default = "conll2012.val.orig_tok_grouped.txt")
	parser.add_argument('--test1', help="Path to test1 orig_tok_grouped.", default = "conll2012.test1.orig_tok_grouped.txt")
	parser.add_argument('--test2', help="Path to test2 orig_tok_grouped (optional).", default = "")
	parser.add_argument('--frameset', help="Path to extracted role set.", default = "frameset.txt")

	parser.add_argument('--output', help="Prefix of the output file names. ", type=str, default = "conll2012")
	opt = parser.parse_args(arguments)

	opt.frameset = opt.dir + opt.frameset
	opt.train = opt.dir + opt.train
	opt.val = opt.dir + opt.val
	opt.test1 = opt.dir + opt.test1
	opt.test2 = opt.dir + opt.test2
	opt.output = opt.dir + opt.output
	opt.roleset_dict = opt.dir + opt.roleset_dict
	opt.label_dict = opt.dir + opt.label_dict

	process(opt)

if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))
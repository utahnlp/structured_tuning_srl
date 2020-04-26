import sys
import os
import argparse
import numpy as np
import h5py
import itertools
from collections import defaultdict
import json
import torch
from torch import cuda
from transformers import *


def get_tokenizer(key):
	model_map={"bert-base-uncased": (BertModel, BertTokenizer),
		"roberta-base": (RobertaModel, RobertaTokenizer)}
	model_cls, tokenizer_cls = model_map[key]
	print('loading tokenizer: {0}'.format(key))
	tokenizer = tokenizer_cls.from_pretrained(key)
	return tokenizer

def get_unk_idx(key):
	unk_map={"bert-base-uncased": 100,
		"roberta-base": 3}
	return unk_map[key]

class Indexer:
	def __init__(self, symbols = ["<blank>"], num_oov=100):
		self.num_oov = num_oov

		self.d = {}
		self.cnt = {}
		for s in symbols:
			self.d[s] = len(self.d)
			self.cnt[s] = 0
			
		for i in range(self.num_oov): #hash oov words to one of 100 random embeddings
			oov_word = '<oov'+ str(i) + '>'
			self.d[oov_word] = len(self.d)
			self.cnt[oov_word] = 10000000	# have a large number for oov word to avoid being pruned
			
	def convert(self, w):		
		return self.d[w] if w in self.d else self.d['<oov' + str(np.random.randint(self.num_oov)) + '>']

	def convert_sequence(self, ls):
		return [self.convert(l) for l in ls]

	def write(self, outfile, with_cnt=True):
		assert(len(self.d) == len(self.cnt))
		with open(outfile, 'w+') as f:
			items = [(v, k) for k, v in self.d.items()]
			items.sort()
			for v, k in items:
				if with_cnt:
					f.write('{0} {1} {2}\n'.format(k, v, self.cnt[k]))
				else:
					f.write('{0} {1}\n'.format(k, v))

	# register tokens only appear in wv
	#   NOTE, only do counting on training set
	def register_words(self, wv, seq, count):
		for w in seq:
			if w in wv and w not in self.d:
				self.d[w] = len(self.d)
				self.cnt[w] = 0
			if w in self.cnt:
				self.cnt[w] = self.cnt[w] + 1 if count else self.cnt[w]

	#   NOTE, only do counting on training set
	def register_all_words(self, seq, count):
		for w in seq:
			if w not in self.d:
				self.d[w] = len(self.d)
				self.cnt[w] = 0
			if w in self.cnt:
				self.cnt[w] = self.cnt[w] + 1 if count else self.cnt[w]

			
def pad(ls, length, symbol, pad_back = True):
	if len(ls) >= length:
		return ls[:length]
	if pad_back:
		return ls + [symbol] * (length -len(ls))
	else:
		return [symbol] * (length -len(ls)) + ls		


def make_vocab(opt, all_word_indexer, label_indexer, roleset_indexer, toks, tok_l, labels, frames, count):
	num_ex = 0
	for _, (toks_orig, tok_l_orig, label_orig, frame_orig) in enumerate(zip(toks, tok_l, labels, frames)):
		toks_orig = toks_orig.strip().split()
		label_orig = label_orig.strip().split()
		tok_l_orig = tok_l_orig.strip().split()
		frame_orig = frame_orig.strip()

		assert(len(tok_l_orig) == len(label_orig))

		all_word_indexer.register_all_words(toks_orig, count)
		label_indexer.register_all_words(label_orig, count)
		roleset_indexer.register_all_words([frame_orig], count)
		num_ex += 1

	return num_ex


def convert(opt, tokenizer, all_word_indexer, label_indexer, roleset_indexer, toks, tok_l, labels, roleset_ids, orig_toks, tokenizer_output, output):
	np.random.seed(opt.seed)

	grouped_input = group_tokenized_with_labels(toks, tok_l, labels, roleset_ids, orig_toks, tokenizer_output)
	num_ex = len(grouped_input)

	tok_idx = np.zeros((num_ex, opt.max_seq_l), dtype=int)
	sub2tok_idx = np.zeros((num_ex, opt.max_seq_l, opt.max_num_subtok), dtype=int) - 1	# empty is -1
	v_idx = np.zeros((num_ex, opt.max_num_v), dtype=int)
	role_label = np.zeros((num_ex, opt.max_num_v, opt.max_seq_l), dtype=int)
	v_length = np.zeros((num_ex,), dtype=int)	# number of v
	v_roleset_id = np.zeros((num_ex, opt.max_num_v), dtype=int)
	prop_idx = np.zeros((num_ex, opt.max_seq_l), dtype=int)
	seq_length = np.zeros((num_ex,), dtype=int)
	orig_seq_length = np.zeros((num_ex,), dtype=int)
	ex_idx = np.zeros(num_ex, dtype=int)
	batch_keys = np.array([None for _ in range(num_ex)])

	ex_id = 0
	for _, (cur_toks, cur_tok_l, cur_list_labels, cur_roleset_ids, cur_orig_toks) in enumerate(grouped_input):
		cur_toks = cur_toks.strip().split()
		cur_tok_l = [int(p) for p in cur_tok_l.strip().split()]
		cur_v_l = len(cur_list_labels)

		tok_idx[ex_id, :len(cur_toks)] = np.array(tokenizer.convert_tokens_to_ids(cur_toks), dtype=int)
		v_length[ex_id] = len(cur_list_labels)
		v_roleset_id[ex_id, :v_length[ex_id]] = roleset_indexer.convert_sequence(cur_roleset_ids)

		# note that the resulted actual seq length after subtoken collapsing could be different even within the same batch
		#	actual seq length is the origial sequence length
		#	seq length is the length after subword tokenization
		acc = 0
		cur_sub2tok = []
		for l in cur_tok_l:
			cur_sub2tok.append(pad([p for p in range(acc, acc+l)], opt.max_num_subtok, -1))
			assert(len(cur_sub2tok[-1]) <= opt.max_num_subtok)
			acc += l
		cur_sub2tok = pad(cur_sub2tok, opt.max_seq_l, [-1 for _ in range(opt.max_num_subtok)])
		sub2tok_idx[ex_id] = np.array(cur_sub2tok, dtype=int)

		orig_seq_length[ex_id] = len(cur_tok_l)
		seq_length[ex_id] = len(cur_toks)
		batch_keys[ex_id] = seq_length[ex_id]

		role_labels = []
		for l_id, cur_labels in enumerate(cur_list_labels):
			cur_labels = cur_labels.strip().split()
			v_idx[ex_id, l_id] = cur_labels.index('B-V')	# there MUST be a B-V since we are reading from role labels

			cur_labels = pad(cur_labels, opt.max_seq_l, 'O')
			cur_labels = label_indexer.convert_sequence(cur_labels)
			role_label[ex_id, l_id] = np.array(cur_labels, dtype=int)

		ex_id += 1
		if ex_id % 10000 == 0:
			print("{}/{} sentences processed".format(ex_id, num_ex))
	
	if opt.shuffle == 1:
		rand_idx = np.random.permutation(ex_id)
		tok_idx = tok_idx[rand_idx]
		v_idx = v_idx[rand_idx]
		role_label = role_label[rand_idx]
		v_length = v_length[rand_idx]
		v_roleset_id = v_roleset_id[rand_idx]
		sub2tok_idx = sub2tok_idx[rand_idx]
		orig_seq_length = orig_seq_length[rand_idx]
		seq_length = seq_length[rand_idx]
		batch_keys = batch_keys[rand_idx]
		ex_idx = rand_idx

	# break up batches based on source/target lengths
	sorted_keys = sorted([(i, p) for i, p in enumerate(batch_keys)], key=lambda x: x[1])
	sorted_idx = [i for i, _ in sorted_keys]
	# rearrange examples	
	tok_idx = tok_idx[sorted_idx]
	v_idx = v_idx[sorted_idx]
	role_label = role_label[sorted_idx]
	v_l = v_length[sorted_idx]
	v_roleset_id = v_roleset_id[sorted_idx]
	sub2tok_idx = sub2tok_idx[sorted_idx]
	seq_l = seq_length[sorted_idx]
	orig_seq_l = orig_seq_length[sorted_idx]
	ex_idx = rand_idx[sorted_idx]

	mark_l = 0
	batch_location = [] #idx where sent length changes
	for j,i in enumerate(sorted_idx):
		if batch_keys[i] != mark_l:
			mark_l = seq_length[i]
			batch_location.append(j)
	if batch_location[-1] != len(tok_idx): 
		batch_location.append(len(tok_idx)-1)
	
	#get batch sizes
	curr_idx = 0
	batch_idx = [0]
	for i in range(len(batch_location)-1):
		end_location = batch_location[i+1]
		while curr_idx < end_location:
			curr_idx = min(curr_idx + opt.batch_size, end_location)
			batch_idx.append(curr_idx)

	batch_l = []
	seq_l_new = []
	for i in range(len(batch_idx)):
		end = batch_idx[i+1] if i < len(batch_idx)-1 else len(tok_idx)
		batch_l.append(end - batch_idx[i])
		seq_l_new.append(seq_l[batch_idx[i]])
		
		# sanity check
		for k in range(batch_idx[i], end):
			assert(seq_l[k] == seq_l_new[-1])
			assert(tok_idx[k, seq_l[k]:].sum() == 0)

	#
	analysis(tok_idx, tokenizer.unk_token_id)

	# Write output
	f = h5py.File(output, "w")		
	f["tok_idx"] = tok_idx
	f["v_idx"] = v_idx
	f["role_label"] = role_label
	f["v_l"] = v_l
	f["v_roleset_id"] = v_roleset_id
	f['sub2tok_idx'] = sub2tok_idx
	f["seq_l"] = np.array(seq_l_new, dtype=int)
	f["orig_seq_l"] = orig_seq_l
	f["batch_l"] = batch_l
	f["batch_idx"] = batch_idx
	f['ex_idx'] = ex_idx
	print("saved {} batches ".format(len(f["batch_l"])))
	f.close()  

def tokenize_and_write(tokenizer, path, output):
	CLS, SEP = tokenizer.cls_token, tokenizer.sep_token
	print('tokenizing sentences from {0}'.format(path))
	all_orig_tok = []
	all_tok = []
	all_tok_l = []
	all_label = []
	all_roleset_ids = []
	act_max_seq_l = 0
	with open(path, 'r') as f:
		for l in f:
			if l.strip() == '':
				continue
			
			parts = l.split('|||')
			assert(len(parts) == 2 or len(parts) == 3)	# if parts has 3: words ||| labels ||| roleset id; if parts has 2: words ||| labels

			if len(parts) == 3:
				roleset_id = parts[-1].strip()
			else:
				roleset_id = '-1'	# -1 indicates no frame

			sent, labels = parts[0].strip().split(), parts[1].strip().split()
			#v_idx = int(sent[0])
			sent = sent[1:]	# removing the first trigger idx
			assert(len(sent) == len(labels))

			sent_subtoks = [tokenizer.tokenize(w) for w in sent]
			tok_l = [len(subtoks) for subtoks in sent_subtoks]
			toks = [p for subtoks in sent_subtoks for p in subtoks]	# flatterning
			orig_toks = [w for w in sent]

			# pad for CLS and SEP
			toks = [CLS] + toks + [SEP]
			labels = ['O'] + labels + ['O']
			tok_l = [1] + tok_l + [1]
			orig_toks = [CLS] + orig_toks + [SEP]

			act_max_seq_l = max(act_max_seq_l, len(toks))

			all_tok.append(' '.join(toks))
			all_tok_l.append(' '.join([str(p) for p in tok_l]))
			all_label.append(' '.join(labels))
			all_orig_tok.append(' '.join(orig_toks))
			all_roleset_ids.append(roleset_id)

	print('act_max_seq_l: {0}'.format(act_max_seq_l))

	print('writing tokenized to {0}'.format(output + '.tok.txt'))
	with open(output + '.tok.txt', 'w') as f:
		for seq in all_tok:
			f.write(seq + '\n')

	print('writing token lengths to {0}'.format(output + '.tok_l.txt'))
	with open(output + '.tok_l.txt', 'w') as f:
		for seq in all_tok_l:
			f.write(seq + '\n')

	print('writing labels to {0}'.format(output + '.label.txt'))
	with open(output + '.label.txt', 'w') as f:
		for seq in all_label:
			f.write(seq + '\n')

	print('writing roleset_ids to {0}'.format(output + '.roleset_id.txt'))
	with open(output + '.roleset_id.txt', 'w') as f:
		for seq in all_roleset_ids:
			f.write(seq + '\n')
			
	print('writing original tokens to {0}'.format(output + '.orig_tok.txt'))
	with open(output + '.orig_tok.txt', 'w') as f:
		for seq in all_orig_tok:
			f.write(seq + '\n')


	return all_tok, all_tok_l, all_label, all_roleset_ids, all_orig_tok


def group_tokenized_with_labels(toks, tok_l, labels, roleset_ids, orig_toks, output):
	act_max_num_v = 0
	sent_map = {}
	for cur_toks, cur_tok_l, cur_labels, cur_roleset_id, cur_orig_toks in zip(toks, tok_l, labels, roleset_ids, orig_toks):
		if cur_toks not in sent_map:
			sent_map[cur_toks] = []
		sent_map[cur_toks].append((cur_tok_l, cur_labels, cur_roleset_id, cur_orig_toks))

	rs = []
	for cur_toks, pairs in sent_map.items():
		act_max_num_v = max(act_max_num_v, len(pairs))
		cur_labels = [p[1] for p in pairs]
		cur_roleset_ids = [p[2] for p in pairs]
		rs.append((cur_toks, pairs[0][0], cur_labels, cur_roleset_ids, pairs[0][3]))

	print('writing grouped original tokens to {0}'.format(output + '.orig_tok_grouped.txt'))
	with open(output + '.orig_tok_grouped.txt', 'w') as f:
		for row in rs:
			f.write(row[-1] + '\n')

	print('act_max_num_v: {0}'.format(act_max_num_v))
	return rs 	# (toks, tok_l, list of labels)


def load(path):
	all_lines = []
	with open(path, 'r') as f:
		for l in f:
			if l.rstrip() == '':
				continue
			all_lines.append(l.strip())
	return all_lines


def load_frameset(path):
	rs = {}
	with open(path, 'r') as f:
		for line in f:
			parts = line.strip().split(' ')
			lemma = parts[0]
			rs[lemma] = []
			if lemma == '#':
				continue
			for p in parts[1:]:
				roleset_id, arg_set = p.split('|')
				rs[lemma].append((roleset_id, [p for p in arg_set.split(',')]))
	return rs


def analysis(toks, unk_idx):
	unk_cnt = 0
	for row in toks:
		if len([1 for idx in row if idx == unk_idx]) != 0:
			unk_cnt += 1
	print('{0} examples has token unk.'.format(unk_cnt))


def process(opt):
	tokenizer = get_tokenizer(opt.bert_type)

	all_word_indexer = Indexer(symbols = ["<blank>", tokenizer.cls_token, tokenizer.sep_token])	# all tokens will be recorded
	label_indexer = Indexer(symbols=["O"], num_oov=0)	# label with O at the 0-th index
	roleset_indexer = Indexer(symbols=["-1"], num_oov=0)	# roleset ids (not lemma, just id), -1 at the 0-th index

	# adding oov words (DEPRECATED)
	oov_words = []
	for i in range(0,100): #hash oov words to one of 100 random embeddings, per Parikh et al. 2016
		oov_words.append('<oov'+ str(i) + '>')
	all_word_indexer.register_all_words(oov_words, count=False)

	# load role set and register to indexer
	frameset = load_frameset(opt.frameset)
	for key, val in frameset.items():
		roleset_indexer.register_all_words([roleset_id for roleset_id, _ in val], count=False)

	#### tokenize
	tokenizer_output = opt.tokenizer_output if opt.tokenizer_output != opt.dir else opt.dir
	train_toks, train_tok_l, train_labels, train_roleset_ids, train_orig_toks = tokenize_and_write(tokenizer, opt.train, tokenizer_output + '.train')
	val_toks, val_tok_l, val_labels, val_roleset_ids, val_orig_toks = tokenize_and_write(tokenizer, opt.val, tokenizer_output + '.val')
	test1_toks, test1_tok_l, test1_labels, test1_roleset_ids, test1_orig_toks = tokenize_and_write(tokenizer, opt.test1, tokenizer_output + '.test1')
	if opt.test2 != opt.dir:
		test2_toks, test2_tok_l, test2_labels, test2_roleset_ids, test2_orig_toks = tokenize_and_write(tokenizer, opt.test2, tokenizer_output + '.test2')

	print("First pass through data to get vocab...")

	num_train = make_vocab(opt, all_word_indexer, label_indexer, roleset_indexer, train_toks, train_tok_l, train_labels, train_roleset_ids, count=True)
	print("Number of examples in training: {}".format(num_train))
	print("Number of sentences in training: {0}, vocab size: {1}".format(num_train, len(all_word_indexer.d)))

	num_val = make_vocab(opt, all_word_indexer, label_indexer, roleset_indexer, val_toks, val_tok_l, val_labels, val_roleset_ids, count=True)
	print("Number of examples in valid: {}".format(num_val))
	print("Number of sentences in valid: {0}, vocab size: {1}".format(num_val, len(all_word_indexer.d))) 

	num_test1 = make_vocab(opt, all_word_indexer, label_indexer, roleset_indexer, test1_toks, test1_tok_l, test1_labels, test1_roleset_ids, count=False)	# no counting on test set
	print("Number of examples in test1: {}".format(num_test1))

	if opt.test2 != opt.dir:
		num_test2 = make_vocab(opt, all_word_indexer, label_indexer, roleset_indexer, test2_toks, test2_tok_l, test2_labels, test2_roleset_ids, count=False)	# no counting on test set
		print("Number of examples in test2: {}".format(num_test2))

	print(label_indexer.d)
	print(roleset_indexer.d)

	all_word_indexer.write(opt.output + ".allword.dict")
	label_indexer.write(opt.output + ".label.dict")
	roleset_indexer.write(opt.output + ".roleset_id.dict")
	print("vocab size: {}".format(len(all_word_indexer.d)))
	print('label size: {}'.format(len(label_indexer.d)))
	print('frame size: {}'.format(len(roleset_indexer.d)))

	convert(opt, tokenizer, all_word_indexer, label_indexer, roleset_indexer, train_toks, train_tok_l, train_labels, train_roleset_ids, train_orig_toks, tokenizer_output+'.train', opt.output + ".train.hdf5")
	convert(opt, tokenizer, all_word_indexer, label_indexer, roleset_indexer, val_toks, val_tok_l, val_labels, val_roleset_ids, val_orig_toks, tokenizer_output+'.val', opt.output + ".val.hdf5")
	convert(opt, tokenizer, all_word_indexer, label_indexer, roleset_indexer, test1_toks, test1_tok_l, test1_labels, test1_roleset_ids, test1_orig_toks, tokenizer_output+'.test1', opt.output + ".test1.hdf5")	
	if opt.test2 != opt.dir:
		convert(opt, tokenizer, all_word_indexer, label_indexer, roleset_indexer, test2_toks, test2_tok_l, test2_labels, test2_roleset_ids, test2_orig_toks, tokenizer_output+'.test2', opt.output + ".test2.hdf5")	
	
	
def main(arguments):
	parser = argparse.ArgumentParser(
		description=__doc__,
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--train', help="Path to training data, sentence and labels are separated by |||.", default = "conll05.train.txt")
	parser.add_argument('--val', help="Path to validation data, sentence and labels are separated by |||.",default = "conll05.devel.txt")
	parser.add_argument('--test1', help="Path to test1 data, sentence and labels are separated by |||.",default = "conll05.test.wsj.txt")
	parser.add_argument('--test2', help="Path to test2 data (optional), sentence and labels are separated by |||.",default = "conll05.test.brown.txt")
	parser.add_argument('--dir', help="Path to the data dir",default = "./data/srl/")
	parser.add_argument('--bert_type', help="The type of bert encoder from huggingface, eg. roberta-base",default = "roberta-base")

	parser.add_argument('--frameset', help="Path to extracted role set.", default = "frameset.txt")
	
	parser.add_argument('--batch_size', help="Size of each minibatch.", type=int, default=48)
	parser.add_argument('--max_seq_l', help="Maximal sequence length", type=int, default=200)
	parser.add_argument('--max_num_v', help="Maximal number of predicate in a sentence", type=int, default=30)
	parser.add_argument('--max_num_subtok', help="Maximal number subtokens in a word", type=int, default=8)
	parser.add_argument('--tokenizer_output', help="Prefix of the tokenized output file names. ", type=str, default = "conll05")
	parser.add_argument('--output', help="Prefix of the output file names. ", type=str, default = "conll05")
	parser.add_argument('--shuffle', help="If = 1, shuffle sentences before sorting (based on source length).", type = int, default = 1)
	parser.add_argument('--seed', help="The random seed", type = int, default = 1)
	opt = parser.parse_args(arguments)

	opt.frameset = opt.dir + opt.frameset
	opt.train = opt.dir + opt.train
	opt.val = opt.dir + opt.val
	opt.test1 = opt.dir + opt.test1
	opt.test2 = opt.dir + opt.test2
	opt.output = opt.dir + opt.output
	opt.tokenizer_output = opt.dir + opt.tokenizer_output

	process(opt)

if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))

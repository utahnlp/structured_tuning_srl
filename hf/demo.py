import sys
import argparse
import h5py
import numpy as np
import torch
from torch import nn
from torch import cuda
from util.holder import *
from util.util import *
from preprocess.preprocess import pad
from hf.roberta_for_srl import *
import traceback

#import spacy
#spacy_nlp = spacy.load('en')
# use nltk instead as it has better token-char mapping
import nltk
from nltk.tokenize import TreebankWordTokenizer
tb_tokenizer = TreebankWordTokenizer()


parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--load_file', help="The path to pretrained model (optional)", default = "")
parser.add_argument('--label_dict', help="The path to label dictionary", default = "./data/srl/conll2012.label.dict")
## pipeline specs
parser.add_argument('--max_num_subtok', help="Maximal number subtokens in a word", type=int, default=8)
# bert specs
parser.add_argument('--bert_type', help="The type of bert encoder from huggingface, eg. roberta-base",default = "roberta-base")
parser.add_argument('--compact_mode', help="How word pieces be mapped to word level label", default='whole_word')
## pipeline stages
parser.add_argument('--enc', help="The type of encoder, e.g., bert", default='bert')
parser.add_argument('--cls', help="The type of classifier, e.g., linear", default='linear')
#
parser.add_argument('--gpuid', help="The GPU index, if -1 then use CPU", type=int, default=-1)
parser.add_argument('--verbose', help="Whether to print out every prediction", type=int, default=0)
parser.add_argument('--num_frame', help="The number of frame for each proposition", type=int, default=38)


def process(opt, tokenizer, seq, predicates):
	bos_tok, eos_tok = get_special_tokens(tokenizer)

	char_spans = list(tb_tokenizer.span_tokenize(seq))
	orig_toks = [seq[s:e] for s, e in char_spans]

	v_label = [next((i for i, span in enumerate(char_spans) if span == (seq.index(p), seq.index(p)+len(p))), None) for p in predicates if p in seq]
	v_label = [i for i in v_label if i is not None]

	if len(v_label) != len(predicates):
		print('valid predicates: ', ','.join([orig_toks[i] for i in v_label]))

	sent_subtoks = [tokenizer.tokenize(t) for t in orig_toks]
	tok_l = [len(subtoks) for subtoks in sent_subtoks]
	toks = [p for subtoks in sent_subtoks for p in subtoks]	# flatterning

	# pad for CLS and SEP
	CLS, SEP = tokenizer.cls_token, tokenizer.sep_token
	toks = [CLS] + toks + [SEP]
	tok_l = [1] + tok_l + [1]
	orig_toks = [CLS] + orig_toks + [SEP]
	v_label = [l+1 for l in v_label]

	tok_idx = np.array(tokenizer.convert_tokens_to_ids(toks), dtype=int)

	# note that the resulted actual seq length after subtoken collapsing could be different even within the same batch
	#	actual seq length is the origial sequence length
	#	seq length is the length after subword tokenization
	acc = 0
	sub2tok_idx = []
	for l in tok_l:
		sub2tok_idx.append(pad([p for p in range(acc, acc+l)], opt.max_num_subtok, -1))
		assert(len(sub2tok_idx[-1]) <= opt.max_num_subtok)
		acc += l
	sub2tok_idx = pad(sub2tok_idx, len(tok_idx), [-1 for _ in range(opt.max_num_subtok)])
	sub2tok_idx = np.array(sub2tok_idx, dtype=int)
	return tok_idx, sub2tok_idx, toks, orig_toks, v_label


def fix_opt(opt):
	opt.loss = 'crf'
	opt.use_gold_predicate = 0
	opt.dropout = 0
	opt.lambd = "1.0"
	opt.param_init_type = 'xavier_uniform'
	return opt


def pretty_print_pred(opt, shared, m, pred_idx):
	batch_l = shared.batch_l
	orig_l = shared.orig_seq_l
	bv_idx = int(np.where(np.asarray(opt.labels) == 'B-V')[0][0])

	pred_log =[]
	for i in range(batch_l):
		orig_l_i = orig_l[i].item()	# convert to scalar
		a_pred_i = pred_idx[i, :orig_l_i, :orig_l_i]

		orig_tok_grouped = shared.res_map['orig_tok_grouped'][i][1:-1]
		pred_log.append(m.crf_loss.compose_log(orig_tok_grouped, a_pred_i[1:-1, 1:-1].cpu(), transpose=False))
	return pred_log


def run(opt, shared, m, tokenizer, seq, predicates=[]):
	tok_idx, sub2tok_idx, toks, orig_toks, v_label = process(opt, tokenizer, seq, predicates)

	m.update_context(orig_seq_l=to_device(torch.tensor([len(orig_toks)]).int(), opt.gpuid), 
		sub2tok_idx=to_device(torch.tensor([sub2tok_idx]).int(), opt.gpuid), 
		res_map={'orig_tok_grouped': [orig_toks]})

	tok_idx = to_device(Variable(torch.tensor([tok_idx]), requires_grad=False), opt.gpuid)

	if len(v_label) != 0:
		v_l = to_device(torch.Tensor([len(v_label)]).long().view(1), opt.gpuid)
		v_label = to_device(torch.Tensor(v_label).long().view(1, -1), opt.gpuid)
	else:
		v_label, v_l = None, None

	with torch.no_grad():
		pred_idx = m.forward(tok_idx, v_label, v_l)

	log = pretty_print_pred(opt, shared, m, pred_idx)[0]
	return orig_toks[1:-1], log


def init(opt):
	opt = fix_opt(opt)
	opt = complete_opt(opt)

	shared = Holder()

	tokenizer = AutoTokenizer.from_pretrained(opt.bert_type, add_special_tokens=False, use_fast=True)
	m = RobertaForSRL.from_pretrained(opt.load_file, global_opt = opt, shared=shared)

	if opt.gpuid != -1:
		m.cuda(opt.gpuid)

	return opt, shared, m, tokenizer


def main(args):
	opt = parser.parse_args(args)
	opt, shared, m, tokenizer = init(opt)

	seq = "He said he knows it."	
	predicates = ['said', 'knows']
	#predicates = []
	orig_toks, log = run(opt, shared, m, tokenizer, seq, predicates)

	print('###################################')
	print('Here is a sample prediction for input:')
	print('>> Input: ', seq)
	print('>> Predicates: ', ','.join(predicates))	# predicates empty
	print(' '.join(orig_toks))
	print(log)

	print('###################################')
	print('#           Instructions          #')
	print('###################################')
	print('>> Enter a input senquence as prompted.')
	print('>> You may also specify ground truth predicates, or leave it empty.')

	while True:
		try:
			print('###################################')
			seq = input("Enter a sequence: ")
			predicates = input('Enter predicates: ')
			predicates = predicates.strip().split(',')

			orig_toks, log = run(opt, shared, m, tokenizer, seq, predicates)
			print(' '.join(orig_toks))
			print(log)

		except KeyboardInterrupt:
			return
		except BaseException as e:
			traceback.print_tb(e.__traceback__)


if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))


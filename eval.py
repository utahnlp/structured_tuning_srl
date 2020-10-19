import sys
from pipeline import *
import argparse
import h5py
import os
import random
import time
import numpy as np
import torch
from torch.autograd import Variable
from torch import nn
from torch import cuda
from holder import *
from data import *

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dir', help="Path to the data dir", default="data/srl/")
parser.add_argument('--data', help="Path to test data hdf5 file.", default="conll05.test.hdf5")
parser.add_argument('--save_file', help="Path to where model to be saved.", default="model")
parser.add_argument('--load_file', help="The path to pretrained model (optional)", default = "")
parser.add_argument('--label_dict', help="The path to label dictionary", default = "conll05.label.dict")
# resource specs
parser.add_argument('--res', help="Path to test resource files, seperated by comma.", default="")
## pipeline specs
parser.add_argument('--max_num_subtok', help="Maximal number subtokens in a word", type=int, default=8)
parser.add_argument('--hidden_size', help="The general hidden size of the pipeline", type=int, default=768)
parser.add_argument('--dropout', help="The dropout probability", type=float, default=0.0)
parser.add_argument('--num_label', help="The number of label", type=int, default=106)
# bert specs
parser.add_argument('--bert_type', help="The type of bert encoder from huggingface, eg. roberta-base",default = "roberta-base")
parser.add_argument('--fix_bert', help="Whether to fix bert update", type=int, default=1)
parser.add_argument('--bert_size', help="The input bert dim", type=int, default=768)
parser.add_argument('--compact_mode', help="How word pieces be mapped to word level label", default='whole_word')
## pipeline stages
parser.add_argument('--enc', help="The type of encoder, bert", default='bert')
parser.add_argument('--cls', help="The type of classifier, linear", default='linear')
parser.add_argument('--loss', help="The type of losses, separated by ,; the first one MUST be role/crf", default='crf')
parser.add_argument('--lambd', help="The weight of losses, separated by ,; ignored if only one loss", default='1.0')
#
parser.add_argument('--gpuid', help="The GPU index, if -1 then use CPU", type=int, default=-1)
parser.add_argument('--seed', help="The random seed", type=int, default=3435)
parser.add_argument('--verbose', help="Whether to print out every prediction", type=int, default=0)
parser.add_argument('--pred_output', help="The prefix to the path of prediction output", default='pred')
parser.add_argument('--param_init_type', help="The type of parameter initialization", default='xavier_normal')
#
parser.add_argument('--use_gold_predicate', help="Whether to use ground truth predicate during evaluation", type=int, default=1)
parser.add_argument('--use_gold_frame', help="Whether to use gold frame for frame_role_loss", type=int, default=1)
parser.add_argument('--conll_output', help="The prefix of conll formated output", default='conll05')
parser.add_argument('--num_frame', help="The number of frame for each proposition", type=int, default=38)
parser.add_argument('--hard_decoding', help="Whether to use hard constraints for viterbi decoding", type=int, default=1)

# the default fwd pass for multiclass loss
def forward_pass(m, tok_idx, batch_ex_idx, batch_l, seq_l, orig_seq_l, sub2tok_idx, res_map):	
	m.update_context(batch_ex_idx, batch_l, seq_l, orig_seq_l, sub2tok_idx, res_map)
	tok_idx = Variable(tok_idx, requires_grad=False)
	output = m.forward(tok_idx)
	return output


def evaluate(opt, shared, m, data):
	m.train(False)
	shared.is_train = False

	val_loss = 0.0
	num_ex = 0
	num_batch = 0

	val_idx, val_num_ex = data.subsample(1.0)
	data_size = val_idx.size()[0]
	print('validating on the {0} batches {1} examples...'.format(data_size, val_num_ex))

	m.begin_pass()
	for i in range(data_size):
		(data_name, tok_idx, batch_ex_idx, batch_l, seq_l, orig_seq_l, sub2tok_idx, v_label, v_l, role_label, v_roleset_id, res_map) = data[i]
		tok_idx = Variable(tok_idx, requires_grad=False)
		v_label = Variable(v_label, requires_grad=False)
		v_l = Variable(v_l, requires_grad=False)
		role_label = Variable(role_label, requires_grad=False)
		v_roleset_id = Variable(v_roleset_id, requires_grad=False)

		with torch.no_grad():
			m.update_context(batch_ex_idx, batch_l, seq_l, orig_seq_l, sub2tok_idx, res_map)
			m.update_loss_context(v_label, v_l, role_label, v_roleset_id)
			batch_loss, pred_idx = m.forward(tok_idx)

		# stats
		val_loss += float(batch_loss.item())
		num_ex += batch_l
		num_batch += 1

	perf, extra_perf = m.get_epoch_metric()
	m.end_pass()
	return (perf, extra_perf, val_loss / num_batch, num_ex)



def main(args):
	opt = parser.parse_args(args)
	shared = Holder()

	# 
	opt.data = opt.dir + opt.data
	opt.res = '' if opt.res == ''  else ','.join([opt.dir + path for path in opt.res.split(',')])
	opt.label_dict = opt.dir + opt.label_dict

	torch.manual_seed(opt.seed)
	if opt.gpuid != -1:
		torch.cuda.set_device(opt.gpuid)
		torch.cuda.manual_seed_all(opt.seed)

	# build model
	m = Pipeline(opt, shared)

	# initializing from pretrained
	m.init_weight()
	print('loading pretrained model from {0}...'.format(opt.load_file))
	param_dict = load_param_dict('{0}.hdf5'.format(opt.load_file))
	m.set_param_dict(param_dict)

	if opt.gpuid != -1:
		m.distribute()	# distribute to multigpu

	# loading data
	res_files = None if opt.res == '' else opt.res.split(',')
	data = Data(opt, opt.data, res_files)

	#
	perf, extra_perf, avg_loss, num_ex = evaluate(opt, shared, m, data)
	extra_perf_str = ' '.join(['{:.4f}'.format(p) for p in extra_perf])
	print('Val {0:.4f} Extra {1} Loss: {2:.4f}'.format(
		perf, extra_perf_str, avg_loss))

	#print('saving model to {0}'.format('tmp'))
	#param_dict = m.get_param_dict()
	#save_param_dict(param_dict, '{0}.hdf5'.format('tmp'))


if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))
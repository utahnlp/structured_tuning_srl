import sys
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
from util.holder import *
from util.util import *
from util.data import *
from modules.optimizer import *
from modules.pipeline import *

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dir', help="Path to the data dir", default="data/srl/")
parser.add_argument('--train_data', help="Path to training data hdf5 file.", default="conll05.train.hdf5")
parser.add_argument('--val_data', help="Path to validation data hdf5 file.", default="conll05.val.hdf5")
parser.add_argument('--save_file', help="Path to where model to be saved.", default="model")
parser.add_argument('--load_file', help="The path to pretrained model (optional)", default = "")
parser.add_argument('--label_dict', help="The path to label dictionary", default = "conll05.label.dict")
# resource specs
parser.add_argument('--train_res', help="Path to training resource files, seperated by comma.", default="")
parser.add_argument('--val_res', help="Path to validation resource files, seperated by comma.", default="")
## pipeline specs
parser.add_argument('--max_num_subtok', help="Maximal number subtokens in a word", type=int, default=8)
parser.add_argument('--hidden_size', help="The general hidden size of the pipeline", type=int, default=768)
parser.add_argument('--dropout', help="The dropout probability", type=float, default=0.1)
parser.add_argument('--percent', help="The percent of training data to use", type=float, default=1.0)
parser.add_argument('--val_percent', help="The percent of validation data to use", type=float, default=1.0)
parser.add_argument('--epochs', help="The number of epoches for training", type=int, default=3)
parser.add_argument('--optim', help="The name of optimizer to use for training", default='adamw_fp16')
parser.add_argument('--learning_rate', help="The learning rate for training", type=float, default=0.001)
parser.add_argument('--clip', help="The norm2 threshold to clip, set it to negative to disable", type=float, default=1.0)
parser.add_argument('--adam_betas', help="The betas used in adam", default='0.9,0.999')
parser.add_argument('--weight_decay', help="The factor of weight decay", type=float, default=0.01)
# bert specs
parser.add_argument('--bert_type', help="The type of bert encoder from huggingface, eg. roberta-base",default = "roberta-base")
parser.add_argument('--warmup_perc', help="The percentages of total expectec updates to warmup", type=float, default=0.1)
parser.add_argument('--compact_mode', help="How word pieces be mapped to word level label", default='whole_word')
## pipeline stages
parser.add_argument('--enc', help="The type of encoder, bert", default='bert')
parser.add_argument('--cls', help="The type of classifier, linear", default='linear')
parser.add_argument('--loss', help="The type of losses, separated by ,; the first one MUST be role/crf", default='crf')
parser.add_argument('--lambd', help="The weight of losses, separated by ,; ignored if only one loss", default='1.0')
#
parser.add_argument('--param_init_type', help="The type of parameter initialization", default='xavier_normal')
parser.add_argument('--print_every', help="Print stats after this many batches", type=int, default=200)
parser.add_argument('--seed', help="The random seed", type=int, default=1)
parser.add_argument('--shuffle_seed', help="The random seed specifically for shuffling", type=int, default=1)
parser.add_argument('--gpuid', help="The GPU index, if -1 then use CPU", type=int, default=-1)
parser.add_argument('--acc_batch_size', help="The accumulative batch size, -1 to disable", type=int, default=-1)
#
parser.add_argument('--use_gold_predicate', help="Whether to use ground truth predicate during evaluation", type=int, default=1)
parser.add_argument('--use_gold_frame', help="Whether to use gold frame for frame_role_loss", type=int, default=1)
parser.add_argument('--conll_output', help="The prefix of conll formated output", default='conll05')
# frame specs
parser.add_argument('--num_frame', help="The number of frame for each proposition", type=int, default=39)


# train batch by batch, accumulate batches until the size reaches acc_batch_size
def train_epoch(opt, shared, m, optim, data, epoch_id, sub_idx):
	train_loss = 0.0
	num_ex = 0
	num_batch = 0
	start_time = time.time()
	num_correct = 0
	min_grad_norm2 = 1000000000000.0
	max_grad_norm2 = 0.0

	# subsamples of data
	# if subsample indices provided, permutate from subsamples
	#	else permutate from all the data
	data_size = sub_idx.size()[0]
	batch_order = torch.randperm(data_size)
	batch_order = sub_idx[batch_order]
	all_data = []
	for i in range(data_size):
		all_data.append((data, batch_order[i]))

	acc_batch_size = 0
	shared.is_train = True
	m.train(True)
	m.begin_pass()
	for i in range(data_size):
		shared.epoch = epoch_id
		shared.has_gold = True
		shared.in_domain = False
		shared.data_size = data_size

		cur_data, cur_idx = all_data[i]
		(data_name, tok_idx, batch_ex_idx, batch_l, seq_l, orig_seq_l, sub2tok_idx, v_label, v_l, role_label, v_roleset_id, res_map) = cur_data[cur_idx]
		tok_idx = Variable(tok_idx, requires_grad=False)
		v_label = Variable(v_label, requires_grad=False)
		v_l = Variable(v_l, requires_grad=False)
		role_label = Variable(role_label, requires_grad=False)
		v_roleset_id = Variable(v_roleset_id, requires_grad=False)

		# fwd pass
		m.update_context(batch_ex_idx, batch_l, seq_l, orig_seq_l, sub2tok_idx, res_map)
		m.update_loss_context(v_label, v_l, role_label, v_roleset_id)
		batch_loss, pred_idx = m.forward(tok_idx)

		# stats
		train_loss += float(batch_loss.item())
		num_ex += batch_l
		num_batch += 1
		time_taken = time.time() - start_time
		acc_batch_size += batch_l

		# accumulate grads
		grad_norm2 = optim.backward(m, batch_loss)

		# accumulate current batch until the rolled up batch size exceeds threshold or meet certain boundary
		if i == data_size-1 or acc_batch_size >= opt.acc_batch_size or (i+1) % opt.print_every == 0:
			optim.step(m)
			shared.num_update += 1

			# clear up grad
			m.zero_grad()
			acc_batch_size = 0

			# stats
			grad_norm2_avg = grad_norm2
			min_grad_norm2 = min(min_grad_norm2, grad_norm2_avg)
			max_grad_norm2 = max(max_grad_norm2, grad_norm2_avg)
			time_taken = time.time() - start_time

			if (i+1) % opt.print_every == 0:
				stats = '{0}, Batch {1:.1f}k '.format(epoch_id+1, float(i+1)/1000)
				stats += 'Grad {0:.1f}/{1:.1f} '.format(min_grad_norm2, max_grad_norm2)
				stats += 'Loss {0:.4f} '.format(train_loss / num_batch)	# just a roughly estimate
				stats += m.print_cur_stats()
				stats += ' Time {0:.1f}'.format(time_taken)
				print(stats)

	perf, extra_perf = m.get_epoch_metric()

	m.end_pass()

	return perf, extra_perf, train_loss / num_ex, num_ex

def train(opt, shared, m, optim, train_data, val_data):
	best_val_perf = -1.0	# something < 0
	test_perf = 0.0
	train_perfs = []
	val_perfs = []
	extra_perfs = []

	# if the specified shuffle seed is not the same as default seed,
	#	force it a random shuffle
	if opt.shuffle_seed != opt.seed:
		torch.manual_seed(opt.seed)

	train_idx, train_num_ex = train_data.subsample(opt.percent)
	print('{0} examples sampled for training'.format(train_num_ex))
	print('for the record, the first 10 training batches are: {0}'.format(train_idx[:10]))
	# sample the same proportion from the dev set as well
	#	but we don't want this to be too small
	minimal_dev_num = max(int(train_num_ex * 0.1), 1000)
	val_idx, val_num_ex = val_data.subsample(opt.val_percent, minimal_num=minimal_dev_num)
	print('{0} examples sampled for dev'.format(val_num_ex))
	print('for the record, the first 10 dev batches are: {0}'.format(val_idx[:10]))

	shared.num_train_ex = train_num_ex
	shared.num_update = 0
	start = 0
	for i in range(start, opt.epochs):
		train_perf, extra_train_perf, loss, num_ex = train_epoch(opt, shared, m, optim, train_data, i, train_idx)
		train_perfs.append(train_perf)
		extra_perf_str = ' '.join(['{:.4f}'.format(p) for p in extra_train_perf])
		print('Train {0:.4f} All {1}'.format(train_perf, extra_perf_str))

		# evaluate
		#	and save if it's the best model
		val_perf, extra_val_perf, val_loss, num_ex = validate(opt, shared, m, val_data, val_idx)
		val_perfs.append(val_perf)
		extra_perfs.append(extra_val_perf)
		extra_perf_str = ' '.join(['{:.4f}'.format(p) for p in extra_val_perf])
		print('Val {0:.4f} All {1}'.format(val_perf, extra_perf_str))

		perf_table_str = ''
		cnt = 0
		print('Epoch  | Train | Val ...')
		for train_perf, extra_perf in zip(train_perfs, extra_perfs):
			extra_perf_str = ' '.join(['{:.4f}'.format(p) for p in extra_perf])
			perf_table_str += '{0}\t{1:.4f}\t{2}\n'.format(cnt+1, train_perf, extra_perf_str)
			cnt += 1
		print(perf_table_str)

		if val_perf > best_val_perf:
			best_val_perf = val_perf
			print('saving model to {0}'.format(opt.save_file))
			param_dict = m.get_param_dict()
			save_param_dict(param_dict, '{0}.hdf5'.format(opt.save_file))
			save_opt(opt, '{0}.opt'.format(opt.save_file))

		else:
			print('skip saving model for perf <= {0:.4f}'.format(best_val_perf))



def validate(opt, shared, m, val_data, val_idx):
	m.train(False)
	shared.is_train = False

	val_loss = 0.0
	num_ex = 0
	num_batch = 0

	data_size = val_idx.size()[0]
	all_val = []
	for i in range(data_size):
		all_val.append((val_data, val_idx[i]))

	#data_size = val_idx.size()[0]
	print('validating on the {0} batches...'.format(data_size))

	m.begin_pass()
	for i in range(data_size):
		cur_data, cur_idx = all_val[i]
		(data_name, tok_idx, batch_ex_idx, batch_l, seq_l, orig_seq_l, sub2tok_idx, v_label, v_l, role_label, v_roleset_id, res_map) = cur_data[cur_idx]
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

	perf, extra_perf = m.get_epoch_metric()	# we only use the first loss's corresponding metric to select models
	m.end_pass()
	return (perf, extra_perf, val_loss / num_batch, num_ex)


def main(args):
	opt = parser.parse_args(args)
	shared = Holder()

	# 
	opt.train_data = opt.dir + opt.train_data
	opt.val_data = opt.dir + opt.val_data
	opt.train_res = '' if opt.train_res == ''  else ','.join([opt.dir + path for path in opt.train_res.split(',')])
	opt.val_res = '' if opt.val_res == ''  else ','.join([opt.dir + path for path in opt.val_res.split(',')])
	opt.label_dict = opt.dir + opt.label_dict

	opt = complete_opt(opt)

	torch.manual_seed(opt.seed)
	if opt.gpuid != -1:
		torch.cuda.set_device(opt.gpuid)
		torch.cuda.manual_seed_all(opt.seed)

	print(opt)

	# build model
	m = Pipeline(opt, shared)
	optim = get_optimizer(opt, shared)

	# initializing from pretrained
	if opt.load_file != '':
		m.init_weight()
		print('loading pretrained model from {0}...'.format(opt.load_file))
		param_dict = load_param_dict('{0}.hdf5'.format(opt.load_file))
		m.set_param_dict(param_dict)
	else:
		m.init_weight()
		model_parameters = filter(lambda p: p.requires_grad, m.parameters())
		num_params = sum([np.prod(p.size()) for p in model_parameters])
		print('total number of trainable parameters: {0}'.format(num_params))
	
	if opt.gpuid != -1:
		m.distribute()	# distribute to multigpu
	m = optim.build_optimizer(m)	# build optimizer after distributing model to devices

	# loading data
	train_res_files = None if opt.train_res == '' else opt.train_res.split(',')
	train_data = Data(opt, opt.train_data, train_res_files)
	val_res_files = None if opt.val_res == '' else opt.val_res.split(',')
	val_data = Data(opt, opt.val_data, val_res_files)

	print('{0} batches in train set'.format(train_data.size()))

	train(opt, shared, m, optim, train_data, val_data)



if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))
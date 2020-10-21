import sys
import h5py
import torch
from torch import nn
from torch import cuda
import string
import re
from collections import Counter
import numpy as np

def get_special_tokens(tokenizer):
	CLS, SEP = tokenizer.cls_token, tokenizer.sep_token
	if CLS is None or SEP is None:
		CLS, SEP = tokenizer.bos_token, tokenizer.eos_token
	if CLS is None:
		CLS = SEP
	return CLS, SEP

def to_device(x, gpuid):
	if gpuid == -1:
		return x.cpu()
	if x.device != gpuid:
		return x.cuda(gpuid)
	return x

def has_nan(t):
	return torch.isnan(t).sum() == 1

def tensor_on_dev(t, is_cuda):
	if is_cuda:
		return t.cuda()
	else:
		return t

def pick_label(dist):
	return np.argmax(dist, axis=1)

def torch2np(t, is_cuda):
	return t.numpy() if not is_cuda else t.cpu().numpy()

def save_opt(opt, path):
	with open(path, 'w') as f:
		f.write('{0}'.format(opt))


def last_index(ls, key):
	return len(ls) - 1 - ls[::-1].index(key)

def load_param_dict(path):
	# TODO, this is ugly
	f = h5py.File(path, 'r')
	return f


def save_param_dict(param_dict, path):
	file = h5py.File(path, 'w')
	for name, p in param_dict.items():
		file.create_dataset(name, data=p)

	file.close()


def load_dict(path):
	rs = {}
	with open(path, 'r+') as f:
		for l in f:
			if l.strip() == '':
				continue
			w, idx, cnt = l.strip().split()
			rs[int(idx)] = w
	return rs


def rand_tensor(shape, r1, r2):
	return (r1 - r2) * torch.rand(shape) + r2


def max_with_mask(v, dim):
	max_v, max_idx = v.max(dim)
	return max_v, max_idx, torch.zeros(v.shape).to(v).scatter(dim, max_idx.unsqueeze(dim), 1.0)

def min_with_mask(v, dim):
	min_v, min_idx = v.min(dim)
	return min_v, min_idx, torch.zeros(v.shape).to(v).scatter(dim, min_idx.unsqueeze(dim), 1.0)


# use the idx (batch_l, seq_l, rs_l) (2nd dim) to select the middle dim of the content (batch_l, seq_l, d)
#	the result has shape (batch_l, seq_l, rs_l, d)
def batch_index2_select(content, idx, nul_idx):
	idx = idx.long()
	rs_l = idx.shape[-1]
	batch_l, seq_l, d = content.shape
	content = content.contiguous().view(-1, d)
	shift = torch.arange(0, batch_l).to(idx.device).long().view(batch_l, 1, 1)
	shift = shift * seq_l
	shifted = idx + shift
	rs = content[shifted].view(batch_l, seq_l, rs_l, d)
	#
	mask = (idx != nul_idx).unsqueeze(-1)
	return rs * mask.to(rs)

# use the idx (batch_l, rs_l) (1st dim) to select the middle dim of the content (batch_l, seq_l, d)
#	return (batch_l, rs_l, d)
def batch_index1_select(content, idx, nul_idx):
	idx = idx.long()
	rs_l = idx.shape[-1]
	batch_l, seq_l, d = content.shape
	content = content.contiguous().view(-1, d)
	shift = torch.arange(0, batch_l).to(idx.device).long().view(batch_l, 1)
	shift = shift * seq_l
	shifted = idx + shift
	rs = content[shifted].view(batch_l, rs_l, d)
	#
	mask = (idx != nul_idx).unsqueeze(-1)
	return rs * mask.to(rs)


# convert role labels into conll format
#	for inconsistent BIO labels, it will hack to make sure of consistency
def convert_role_labels(labels):
	inconsistent_cnt = 0
	rs = []
	cur_label = None
	for i, l in enumerate(labels):
		if l.startswith('B-'):
			if cur_label is not None:
				rs[-1] = '*)' if not rs[-1].startswith('(') else rs[-1] + ')'
			rs.append('({0}*'.format(l[2:]))
			cur_label = l[2:]
		elif l.startswith('I-'):
			if cur_label is not None:
				# if there is a inconsistency in label dependency, we fix it by treating this label as B- and end the previous label
				# 	this is a safeguard just in case, because the srl-eval.pl doesn't accept violation on that
				if cur_label != l[2:]:
					inconsistent_cnt += 1
					# take this label as B- then
					rs[-1] = '*)' if not rs[-1].startswith('(') else rs[-1] + ')'
					rs.append('({0}*'.format(l[2:]))
					#raise Exception('inconsistent labels: {0} {1}'.format(cur_l, l))
				else:
					rs.append('*' if i != len(labels)-1 else '*)')
			else:
				# take this label as B- then
				rs.append('({0}*'.format(l[2:]))
				#raise Exception('inconsistent labels: {0} {1}'.format(cur_l, l))
			cur_label = l[2:]
		elif l == 'O':
			if cur_label is not None:
				rs[-1] = '*)' if not rs[-1].startswith('(') else rs[-1] + ')'
			rs.append('*')
			cur_label = None
		else:
			raise Exception('unrecognized label {0}'.format(l))
	return rs, inconsistent_cnt


def system_call_eval(gold_path, pred_path):
	import subprocess
	rs = subprocess.check_output(['perl', 'srl-eval.pl', gold_path, pred_path])
	target_line = rs.decode('utf-8').split('\n')[6].split()
	f1 = float(target_line[-1])*0.01	# make percent to [0,1]
	return f1


if __name__ == '__main__':
	labels = 'O B-V I-V B-A0 B-A1 I-A1 O B-A3 I-A2 I-A2'.split()
	rs, _ = convert_role_labels(labels)
	print(labels)
	print(rs)
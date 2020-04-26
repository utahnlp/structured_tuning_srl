import sys
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from holder import *
from util import *

# DEPRECATED
# predicate-argument Loss, standard nll loss
class RoleLoss(torch.nn.Module):
	def __init__(self, opt, shared):
		super(RoleLoss, self).__init__()
		self.opt = opt
		self.shared = shared
		
		self.non_O_acc_sum = 0.0
		self.num_ex = 0
		self.verbose = False

		self.gold_log = []
		self.pred_log = []

		self.labels = []
		with open(self.opt.label_dict, 'r') as f:
			for l in f:
				if l.strip() == '':
					continue
				toks = l.rstrip().split()
				self.labels.append(toks[0])
		self.labels = np.asarray(self.labels)
		

	def forward(self, log_pa, score, v_label, v_l, role_label, roleset_id, extra={}):
		# in case of using fp16, we want the loss to be in fp32
		log_pa = log_pa.cpu().float()

		batch_l, source_l, _, _ = log_pa.shape
		orig_l = self.shared.orig_seq_l
		max_orig_l = orig_l.max()

		a_loss = torch.zeros(1)
		num_prop = 0
		for i in range(batch_l):
			a_pred_i = log_pa[i, :orig_l[i], :orig_l[i], :]
			v_i = v_label[i, :v_l[i]]
			role_i = role_label[i, :v_l[i], :orig_l[i]]	# (num_v, orig_l)
			a_gold_i = torch.zeros(orig_l[i], orig_l[i]).long()	# O has idx 0

			for k, role_k in enumerate(role_i):
				a_gold_i[v_i[k]] = role_k

			loss_i = -a_pred_i.gather(2, a_gold_i.unsqueeze(-1)).squeeze(-1)	# (orig_l, orig_l)

			if self.opt.use_gold_predicate == 1:
				mask_i = torch.zeros(orig_l[i], orig_l[i])
				for k in range(v_l[i]):
					mask_i[v_i[k], :orig_l[i]] = 1.0
				loss_i = loss_i * mask_i

			a_loss = a_loss + loss_i.sum()
			num_prop += v_l[i]

		self.analyze(log_pa.data, v_label, v_l, role_label)
		self.num_ex += batch_l

		# # average over number of predicates or num_ex
		normalizer = float(num_prop) if self.opt.use_gold_predicate == 1 else sum([orig_l[i] for i in range(batch_l)])
		pred_idx = log_pa[:, :max_orig_l, :max_orig_l, :].argmax(-1)

		return a_loss / normalizer, pred_idx


	def analyze(self, log_pa, v_label, v_l, role_label):
		batch_l, source_l, _, _ = log_pa.shape
		orig_l = self.shared.orig_seq_l
		pred = np.argmax(log_pa.numpy(), axis=3)
		bv_idx = int(np.where(self.labels == 'B-V')[0][0])

		for i in range(batch_l):
			orig_l_i = orig_l[i].item()	# convert to scalar
			v_i = v_label[i, :v_l[i]]
			role_i = role_label[i, :v_l[i], :orig_l_i]	# (num_v, orig_l)

			a_gold_i = torch.zeros(orig_l_i, orig_l_i).long()	# O has idx 0
			for k, role_k in enumerate(role_i):
				a_gold_i[v_i[k]] = role_k

			a_pred_i = np.argmax(log_pa[i, :orig_l_i, :orig_l_i, :], axis=2)
			# if using gold predicate during evaluation, wipe out non-predicate predictions
			if self.opt.use_gold_predicate == 1:
				a_pred_i_new = torch.zeros(orig_l_i, orig_l_i).long()	# O has idx 0
				for k, _ in enumerate(role_i):
					a_pred_i_new[v_i[k]] = a_pred_i[v_i[k]]
					a_pred_i_new[v_i[k], v_i[k]] = bv_idx
				a_pred_i = a_pred_i_new

			non_O_overlap = 0.0
			non_O_cnt = 0
			for k in range(orig_l_i):
				if (a_pred_i[k] != 0).sum() > 0 or (a_gold_i[k] != 0).sum() > 0:
					non_O_overlap += (a_pred_i[k] == a_gold_i[k]).sum()
					non_O_cnt += 1

			self.non_O_acc_sum += (float(non_O_overlap) / (non_O_cnt*orig_l_i)) if non_O_cnt != 0 else 0.0

			# do analysis without cls and sep
			if not self.shared.is_train:
				orig_tok_grouped = self.shared.res_map['orig_tok_grouped'][i][1:-1]
				self.gold_log.append(self.compose_log(orig_tok_grouped, a_gold_i[1:-1, 1:-1]))
				self.pred_log.append(self.compose_log(orig_tok_grouped, a_pred_i[1:-1, 1:-1]))


	# return a string of stats
	def print_cur_stats(self):
		stats = 'Non-O acc {:.3f} '.format(self.non_O_acc_sum / self.num_ex)
		return stats

	# get training metric (scalar metric, extra metric)
	def get_epoch_metric(self):
		non_O_acc = self.non_O_acc_sum / self.num_ex
		return non_O_acc, [non_O_acc] 	# and any other scalar metrics	

	# compose log for one example
	#	role_labels of shape (seq_l, seq_l)
	def compose_log(self, orig_toks, role_labels):
		role_labels = role_labels.numpy()
		seq_l = role_labels.shape[0]

		header = ['-' for _ in range(seq_l)]
		role_lines = []
		for row in role_labels:
			roles = self.labels[row].tolist()
			roles = roles + ['O']	# TODO, the convert_role_labels prefers the last label to be O, so bit hacky here
			if 'B-V' in roles:
				v_idx = roles.index('B-V')
				header[v_idx] = orig_toks[v_idx]
				roles = convert_role_labels(roles)
				role_lines.append(roles[:-1])

		log = [header] + role_lines
		log = np.asarray(log)
		# do a transpose
		log = log.transpose((1, 0))
		
		rs = []
		for row in log:
			rs.append(' '.join(row))
		return '\n'.join(rs) + '\n'


	def begin_pass(self):
		# clear stats
		self.non_O_acc_sum = 0.0
		self.num_ex = 0
		self.gold_log = []
		self.pred_log = []

	def end_pass(self):
		if not self.shared.is_train:
			print('writing gold to {}'.format(self.opt.conll_output + '.gold.txt'))
			with open(self.opt.conll_output + '.gold.txt', 'w') as f:
				for ex in self.gold_log:
					f.write(ex + '\n')
	
			print('writing pred to {}'.format(self.opt.conll_output + '.pred.txt'))
			with open(self.opt.conll_output + '.pred.txt', 'w') as f:
				for ex in self.pred_log:
					f.write(ex + '\n')

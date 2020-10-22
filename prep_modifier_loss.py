import sys
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from holder import *
from util import *

# if an argument is LOC/TMP for a predicate, then it can only be LOC/TMP or O for any other predicate
class PrepModifierLoss(torch.nn.Module):
	def __init__(self, opt, shared):
		super(PrepModifierLoss, self).__init__()
		self.opt = opt
		self.shared = shared
		
		self.one = torch.ones(1).float()
		if opt.gpuid != -1:
			self.one = to_device(self.one, self.opt.gpuid)

		self.violation_cnt = 0
		self.num_prop = 0
		self.num_ex = 0
		
		self.prep_labels = []
		self.prep_label_idx = []
		self.o_idx = 0
		for idx, l in self.opt.label_map_inv.items():
			if l.startswith('B-A'):
				if l.endswith('LOC') or l.endswith('TMP'):
					self.prep_labels.append(l)
					self.prep_label_idx.append(idx)
		print('preposition modifier constraint applies to: ', self.prep_labels)


	def forward(self, log_pa, score, v_label, v_l, role_label, roleset_id, extra):
		orig_l = self.shared.orig_seq_l
		batch_l, source_l, _, _ = log_pa.shape
		num_prep = len(self.prep_label_idx)

		assert(self.opt.use_gold_predicate == 1)
		loss = torch.zeros(1)
		if self.opt.gpuid != -1:
			loss = to_device(loss, self.opt.gpuid)

		#self._gold_role = role_label

		num_prop = 0
		for i in range(batch_l):
			v_i = v_label[i, :v_l[i]]

			if self.opt.use_gold_predicate == 1:
				log_prep = log_pa[i, v_i, :orig_l[i]][:, :, self.prep_label_idx]	# num_v, orig_l, num_prep
				log_o = log_pa[i, v_i, :orig_l[i]][:, :, self.o_idx:self.o_idx+1]	# num_v, orig_l, 1
				log_o = log_o.expand(-1, -1, num_prep)

				rhs = torch.max(log_prep, log_o)	# num_v, orig_l, num_prep
				rhs = rhs.unsqueeze(0).expand(v_l[i], -1, -1, -1)	# num_v, num_v, orig_l, num_prep
				mask = torch.eye(v_l[i]).to(rhs)*10000
				rhs = rhs + mask.unsqueeze(-1).unsqueeze(-1)				

				loss_i = torch.relu(log_prep - rhs.min(1)[0]).sum()

			loss = loss + loss_i
			num_prop += v_l[i]

		# stats
		if hasattr(self.shared, 'viterbi_pred'):
			pred_idx = self.shared.viterbi_pred
		else:
			pred_idx = log_pa.argmax(-1)
		self.analyze(pred_idx, v_label, v_l)

		# stats
		self.num_prop += int(num_prop)
		self.num_ex += batch_l

		# # average over number of predicates or num_ex
		normalizer = float(num_prop) if self.opt.use_gold_predicate == 1 else sum([orig_l[i] for i in range(batch_l)])
		#print('framrrole', loss / normalizer)
		return loss / normalizer, None

	def analyze(self, pred_idx, v_label, v_l):
		batch_l, source_l, _ = pred_idx.shape
		for i in range(batch_l):
			v_i = v_label[i, :v_l[i]]
			pred_idx_i = pred_idx[i, v_i, :]	# (num_v, source_l)
			#pred_idx_i = self._gold_role[i, :v_l[i]]

			mask_o = pred_idx_i == self.o_idx
			sum_o = mask_o.int().sum(0)
			for prep_label_idx in self.prep_label_idx:
				mask_p = pred_idx_i == prep_label_idx
				sum_p = mask_p.int().sum(0)
				violation = (sum_p > 0) * ((sum_o + sum_p)  < v_l[i].item())
				if violation.sum() > 0:
					self.violation_cnt += 1


	# return a string of stats
	def print_cur_stats(self):
		rho = float(self.violation_cnt) / self.num_ex
		return "prep_modifier rho: {0:.3f}".format(rho)
		#return ""

	# get training metric (scalar metric, extra metric)
	def get_epoch_metric(self):
		rho = float(self.violation_cnt) / self.num_ex
		return rho, [rho]
		#return None, []


	def begin_pass(self):
		self.violation_cnt = 0
		self.num_prop = 0
		self.num_ex = 0

	def end_pass(self):
		pass

import sys
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from holder import *
from util import *

# loss on unique core argument
class UniqueRoleLoss(torch.nn.Module):
	def __init__(self, opt, shared):
		super(UniqueRoleLoss, self).__init__()
		self.opt = opt
		self.shared = shared
		
		self.one = torch.ones(1).float()
		if opt.gpuid != -1:
			self.one = to_device(self.one, self.opt.gpuid)

		self.violation_cnt = 0
		self.num_prop = 0
		self.num_ex = 0

		self.core_labels = []
		self.core_label_idx = []
		for idx, l in self.opt.label_map_inv.items():
			if 'B-A' in l and l[-1].isnumeric():	# only applies to B-A*
				self.core_labels.append(l)
				self.core_label_idx.append(idx)
		print('unique role constraint applies to: ', self.core_labels)


	def forward(self, log_pa, score, v_label, v_l, role_label, roleset_id, extra):
		orig_l = self.shared.orig_seq_l
		batch_l, source_l, _, _ = log_pa.shape

		assert(self.opt.use_gold_predicate == 1)
		loss = torch.zeros(1)
		if self.opt.gpuid != -1:
			loss = to_device(loss, self.opt.gpuid)

		#self._gold_role = role_label

		num_prop = 0
		for i in range(batch_l):
			v_i = v_label[i, :v_l[i]]

			if self.opt.use_gold_predicate == 1:
				log_core = log_pa[i, v_i, :orig_l[i]][:, :, self.core_label_idx]	# num_v, orig_l, num_core_label
				log_core_ext = log_core.unsqueeze(1).expand(-1, orig_l[i], -1, -1)	# num_v, orig_l, orig_l, num_core_label

				mask = torch.eye(orig_l[i]).to(log_core)*10000
				log_neg_core_ext = (self.one - log_core_ext.exp()).clamp(min=1e-6).log()	# num_v, orig_l, orig_l, num_core_label
				log_neg_core_ext = log_neg_core_ext + mask.unsqueeze(0).unsqueeze(-1)

				loss_i = torch.relu(log_core - log_neg_core_ext.min(2)[0]).sum()

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
			acc = None
			for k, l in enumerate(self.core_label_idx):
				dup_core_cnt = ((pred_idx_i == l).sum(-1) > 1).int()	# (num_v)
				acc = dup_core_cnt if k == 0 else acc + dup_core_cnt
			self.violation_cnt += (acc > 0).sum().item()


	# return a string of stats
	def print_cur_stats(self):
		rho = float(self.violation_cnt) / self.num_prop
		return "unique rho: {0:.3f}".format(rho)
		#return ""

	# get training metric (scalar metric, extra metric)
	def get_epoch_metric(self):
		rho = float(self.violation_cnt) / self.num_prop
		return rho, [rho]
		#return None, []


	def begin_pass(self):
		self.violation_cnt = 0
		self.num_prop = 0
		self.num_ex = 0

	def end_pass(self):
		pass

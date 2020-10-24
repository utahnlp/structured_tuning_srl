import sys
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from util.holder import *
from util.util import *

# if there is an B-C-x, then there must be a B-x before it
class ContinuousRoleLoss(torch.nn.Module):
	def __init__(self, opt, shared):
		super(ContinuousRoleLoss, self).__init__()
		self.opt = opt
		self.shared = shared
		
		self.one = torch.ones(1).float()
		if opt.gpuid != -1:
			self.one = to_device(self.one, self.opt.gpuid)

		self.violation_cnt = 0
		self.num_prop = 0
		self.num_ex = 0
		
		self.covered_labels = []
		self.covered_b_idx = []
		self.covered_bc_idx = []
		self.label_map = {idx: l for l, idx in self.opt.label_map_inv.items()}
		for l, idx in self.label_map.items():
			if l.startswith('B-C-'):
				self.covered_labels.append(l[4:])
				self.covered_b_idx.append(self.label_map['B-' + l[4:]])
				self.covered_bc_idx.append(idx)
		print('continuous role constraint applies to: ', self.covered_labels)


	def forward(self, log_pa, score, v_label, v_l, role_label, roleset_id, extra):
		orig_l = self.shared.orig_seq_l
		batch_l, source_l, _, _ = log_pa.shape
		num_covered_label = len(self.covered_labels)

		assert(self.opt.use_gold_predicate == 1)
		loss = torch.zeros(1)
		if self.opt.gpuid != -1:
			loss = to_device(loss, self.opt.gpuid)

		#self._gold_role = role_label

		num_prop = 0
		for i in range(batch_l):
			v_i = v_label[i, :v_l[i]]

			if self.opt.use_gold_predicate == 1:
				log_pa_i = log_pa[i, v_i, :orig_l[i]]
				log_b = log_pa_i[:, :, self.covered_b_idx]		# num_v, orig_l, num_covered_label
				log_bc = log_pa_i[:, :, self.covered_bc_idx]	# num_v, orig_l, num_covered_label

				rhs = log_bc.unsqueeze(2).expand(-1, -1, orig_l[i], -1)	# num_v, orig_l, orig_l, num_covered_label

				mask = torch.triu(torch.ones(orig_l[i], orig_l[i]).to(log_pa)).unsqueeze(0).unsqueeze(-1)
				rhs = (rhs + mask * -1e8).max(2)[0]

				loss_i = torch.relu(log_b - rhs).sum()

			loss = loss + loss_i
			num_prop += v_l[i]

		# stats
		if hasattr(self.shared, 'viterbi_pred'):
			pred_idx = self.shared.viterbi_pred
		else:
			pred_idx = log_pa.argmax(-1)
		self.analyze(pred_idx.data.cpu(), v_label, v_l)

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

			for v in range(v_l[i]):
				violated = False
				for b, bc in zip(self.covered_b_idx, self.covered_bc_idx):
					ls = np.where(pred_idx_i[v] == bc)[0]
					for k in ls:
						if len(np.where(pred_idx_i[v, :k] == b)[0]) == 0:
							self.violation_cnt += 1
							violated = True
							break
					if violated:
						break


	# return a string of stats
	def print_cur_stats(self):
		rho = float(self.violation_cnt) / self.num_prop
		return "continuous rho: {0:.3f}".format(rho)
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

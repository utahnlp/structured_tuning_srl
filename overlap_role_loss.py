import sys
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from holder import *
from util import *

# loss on overlap core argument across multiple predicates
class OverlapRoleLoss(torch.nn.Module):
	def __init__(self, opt, shared):
		super(OverlapRoleLoss, self).__init__()
		self.opt = opt
		self.shared = shared
		
		self.one = torch.ones(1).float()
		if opt.gpuid != -1:
			self.one = to_device(self.one, self.opt.gpuid)

		self.violation_ex_cnt = 0
		self.violation_cnt = 0
		self.grad_coverage_cnt = 0
		self.num_prop = 0
		self.num_ex = 0
		self.loss_acc = 0.0
		
		self.covered_labels = {}
		self.covered_label_idx_b = []
		self.covered_label_idx_i = []
		for l, idx in self.opt.labels.items():
			if l.startswith('B-') and ('I-' + l[2:]) in self.opt.labels:
				self.covered_labels[l[2:]] = [idx, self.opt.labels['I-' + l[2:]]]
		for l, (idx1, idx2) in self.covered_labels.items():
			self.covered_label_idx_b.append(idx1)
			self.covered_label_idx_i.append(idx2)
		print('overlap role constraint applies to: ', self.covered_labels)


	def forward(self, log_pa, score, v_label, v_l, role_label, roleset_id, extra):
		orig_l = self.shared.orig_seq_l
		batch_l, source_l, _, _ = log_pa.shape
		num_covered_label = len(self.covered_label_idx_b)

		assert(self.opt.use_gold_predicate == 1)
		loss = torch.zeros(1)
		if self.opt.gpuid != -1:
			loss = to_device(loss, self.opt.gpuid)


		#self._gold_role = role_label

		num_prop = 0
		for i in range(batch_l):
			v_i = v_label[i, :v_l[i]]

			if self.opt.use_gold_predicate == 1:
				log_role = log_pa[i, v_i, :orig_l[i]]
				log_core_b = log_role[:, :, self.covered_label_idx_b]	# num_v, orig_l, num_covered_label
				log_core_i = log_role[:, :, self.covered_label_idx_i]	# num_v, orig_l, num_covered_label
				log_neg_core_i = (self.one - log_core_i.exp()).clamp(min=1e-6).log()

				log_ix_and_nix = torch.min(log_core_i[:, 1:-1, :], log_neg_core_i[:, 2:, :])	# num_v, orig_l-2, num_covered_label
				log_ix_and_nix_ext = log_ix_and_nix.unsqueeze(1).expand(-1, orig_l[i]-2, -1, -1)	# num_v, orig_l-2, orig_l-2, num_covered_label
				log_core_b_ext = log_core_b[:, :-2, :].unsqueeze(2).expand(-1, -1, orig_l[i]-2, -1)	# num_v, orig_l-2, orig_l-2, num_covered_label

				# compute the lhs
				# num_v, orig_l-2, orig_l-2, num_covered_label
				lhs = torch.min(log_core_b_ext, log_ix_and_nix_ext)
				mask = torch.tril(torch.ones(orig_l[i]-2, orig_l[i]-2), diagonal=-1).to(lhs).unsqueeze(0).unsqueeze(-1)
				lhs = lhs + mask * -1e8	# set the lower triangle to neg inf (diagonal not affected!)
				#	get the max "joint probabilities"

				loss_i = torch.zeros(1)
				if self.opt.gpuid != -1:
					loss_i = to_device(loss_i, self.opt.gpuid)

				topk = min(4, (orig_l[i]-2)**2)	# TODO, k is hardcoded here
				lhs_flat = lhs.view(v_l[i], (orig_l[i]-2)**2, num_covered_label)
				lhs_topk, lhs_idx = lhs_flat.topk(topk, dim=1)
				for k in range(topk):
					lhs = lhs_topk[:, k, :].contiguous()
					lhs_mask = torch.zeros(lhs_flat.shape).to(lhs_flat).scatter(1, lhs_idx[:, k:k+1, :], 1.0)

					# exclusion 1
					log_by_and_iy = torch.min(log_core_b[:, 1:-1, :], log_core_i[:, 2:, :])
					cond1 = (self.one - log_by_and_iy.exp()).clamp(min=1e-6).log()	# num_v, orig_l-2, num_covered_label
					cond1 = cond1.unsqueeze(1).expand(-1, orig_l[i]-2, -1, -1)

					# exclusion 2
					log_by_or_iy = torch.max(log_core_b[:, :-2, :], log_core_i[:, :-2, :])	# num_v, orig_l-2, num_covered_label
					log_not_iy_or_not_iy = torch.max(log_neg_core_i[:, 1:-1,: ], log_neg_core_i[:, 2:, :])	# num_v, orig_l-2, num_covered_label
					log_not_iy_or_not_iy_ext = log_not_iy_or_not_iy.unsqueeze(1).expand(-1, orig_l[i]-2, -1, -1)
					log_by_or_iy_ext = log_by_or_iy.unsqueeze(2).expand(-1, -1, orig_l[i]-2, -1)
					cond2 = torch.max(log_by_or_iy_ext, log_not_iy_or_not_iy_ext)

					# rhs
					rhs = torch.min(cond1, cond2).view(v_l[i], -1, num_covered_label)	# num_v, (orig_l-2)^2, num_covered_label
					rhs = rhs * lhs_mask + (self.one - lhs_mask) * 1e8
					rhs = rhs.min(1)[0]	# num_v, num_covered_label

					# loss
					rhs = rhs.view(-1).unsqueeze(0).expand(v_l[i]*num_covered_label, -1)
					mask = torch.eye(v_l[i]*num_covered_label).to(rhs) * 1e8
					rhs = rhs + mask
					loss_i = loss_i + torch.relu(lhs.view(-1) - rhs.min(1)[0]).sum()
			
			# TODO, loss might need to be averaged by k
			loss = loss + loss_i
			num_prop += v_l[i]

			#spans = self.get_spans(log_pa.argmax(-1)[i, v_i, :orig_l[i]].data.cpu())
			#cnt = self.count_overlap(spans)
			#if cnt != 0:
			#	print(loss_i.data.item(), cnt)

			if loss_i.data.item() != 1e-4:
				self.grad_coverage_cnt += 1


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
		orig_l = self.shared.orig_seq_l
		batch_l, source_l, _, = pred_idx.shape
		for i in range(batch_l):
			v_i = v_label[i, :v_l[i]]
			spans = self.get_spans(pred_idx[i, v_i, :orig_l[i]].data.cpu())
			cnt = self.count_overlap(spans)
			self.violation_cnt += cnt
			self.violation_ex_cnt += 1 if cnt > 0 else 0


	def get_spans(self, labels):
		rs = []
		num_v, orig_l = labels.shape
		for v in range(num_v):
			for lb, li in zip(self.covered_label_idx_b, self.covered_label_idx_i):
				start = np.where(labels[v] == lb)[0]
				for s in start:
					end = np.where(labels[v, s+1:] != li)[0]
					e = end[0]+s if len(end) != 0 else s
					rs.append((s, e))
		return rs

	def count_overlap(self, spans):
		cnt = 0
		spans = sorted(spans)
		for i in range(len(spans)):
			l = spans[i]
			for j in range(i, len(spans)):
				r = spans[j]
				if l[0] < r[0] and l[1] > r[0] and l[1] < r[1]:
					cnt += 1
		return cnt

	# return a string of stats
	def print_cur_stats(self):
		#if not self.shared.is_train:
		rho = float(self.violation_ex_cnt) / self.num_ex
		return "overlap rho: {0:.3f} overlap {1}".format(rho, self.violation_cnt)
		#return "overlap loss: {0:.3f}".format(float(self.loss_acc) / self.num_ex)
		#return ""

	# get training metric (scalar metric, extra metric)
	def get_epoch_metric(self):
		print('overlap: {0}'.format(self.violation_cnt))
		print('overlap grad coverage: {0:.3f}'.format(float(self.grad_coverage_cnt)/self.num_ex))
		rho = float(self.violation_ex_cnt) / self.num_ex
		return rho, [rho, self.violation_cnt]
		#loss_avg = float(self.loss_acc) / self.num_ex
		#return loss_avg, [loss_avg]


	def begin_pass(self):
		self.violation_ex_cnt = 0
		self.violation_cnt = 0
		self.grad_coverage_cnt = 0
		self.num_prop = 0
		self.num_ex = 0
		self.loss_acc = 0.0

	def end_pass(self):
		pass

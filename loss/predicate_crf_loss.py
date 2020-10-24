import sys
import torch
from torch import nn
from torch.autograd import Variable
from util.holder import *
from util.util import *
from .crf import *
import time

# CRF loss function incl. decoding
#	only for predicate
class PredicateCRFLoss(torch.nn.Module):
	def __init__(self, opt, shared):
		super(PredicateCRFLoss, self).__init__()
		self.opt = opt
		self.shared = shared

		self.labels = np.asarray(self.opt.labels)

		constraints = allowed_transitions("BIO", self.opt.label_map_inv)

		self.crf = ConditionalRandomField(num_tags=opt.num_label, constraints=constraints, gpuid=opt.gpuid)

		self.quick_acc_sum = 0.0
		self.num_ex = 0

	def decode(self, log_p, score):
		batch_l, source_l, _ = score.shape
		orig_l = self.shared.orig_seq_l
		max_orig_l = orig_l.max()

		score = score[:, :max_orig_l]

		v_mask = torch.zeros(batch_l, max_orig_l).byte()
		for i in range(batch_l):
			v_mask[i, :orig_l[i]] = True
		v_mask = to_device(v_mask, self.opt.gpuid)

		decoded = self.crf.viterbi_tags(score, v_mask)

		pred_idx = torch.zeros(batch_l, max_orig_l).long()
		for i in range(batch_l):
			pred_idx[i, :orig_l[i]] = torch.Tensor(decoded[i][0]).long()
		pred_idx = to_device(pred_idx, self.opt.gpuid)

		return pred_idx, {}


	def forward(self, log_p, score, v_label, v_l, role_label, roleset_id, extra={}):
		batch_l, source_l, _ = score.shape
		orig_l = self.shared.orig_seq_l
		max_orig_l = orig_l.max()

		bv_idx = int(np.where(self.labels == 'B-V')[0][0])
		score = score[:, :max_orig_l]

		assert(self.opt.use_gold_predicate == 1)
		bv_mask = (role_label == bv_idx).sum(1) > 0	# batch_l, max_orig_l
		v_gold = bv_mask.long() * bv_idx
		v_mask = torch.zeros(batch_l, max_orig_l).byte()
		for i in range(batch_l):
			v_mask[:, :orig_l[i]] = True
		v_gold = to_device(v_gold, self.opt.gpuid)
		v_mask = to_device(v_mask, self.opt.gpuid)

		if self.shared.is_train:
			loss = self.crf(score, v_gold, v_mask)

			pred_idx = log_p[:, :max_orig_l, :].argmax(-1)
	
		else:
			# during evaluation, no need to compute the loss, just =0
			loss = to_device(torch.zeros(1), self.opt.gpuid)

			pred_idx, _ = self.decode(log_p, score)

		correct_cnt = (pred_idx > 0).logical_and(v_gold > 0).sum(-1)
		denom_cnt = (pred_idx > 0).logical_or(v_gold > 0).sum(-1)
		self.quick_acc_sum += (correct_cnt.float() / denom_cnt.float()).sum()

		self.num_ex += batch_l
		self.shared.viterbi_pred = pred_idx

		return loss / batch_l, pred_idx

	# return a string of stats
	def print_cur_stats(self):
		stats = 'Quick acc {:.3f}'.format(self.quick_acc_sum / self.num_ex)
		return stats

	# get training metric (scalar metric, extra metric)
	def get_epoch_metric(self):
		quick_acc = self.quick_acc_sum / self.num_ex
		return quick_acc, [quick_acc] 	# and any other scalar metrics	

	def begin_pass(self):
		# clear stats
		self.quick_acc_sum = 0
		self.num_ex = 0

	def end_pass(self):
		pass

if __name__ == '__main__':
	pass





		
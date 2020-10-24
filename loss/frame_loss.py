import sys
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from util.holder import *
from util.util import *

# loss on frame id prediction
class FrameLoss(torch.nn.Module):
	def __init__(self, opt, shared):
		super(FrameLoss, self).__init__()
		self.opt = opt
		self.shared = shared
		self.num_correct = 0
		self.num_ex = 0
		self.num_prop = 0
		

	def forward(self, log_pa, score, v_label, v_l, role_label, roleset_id, extra):
		assert(len(extra) != 0)
		log_frame = extra['frame']	# batch_l, source_l, num_frame
		frame_idx = self.shared.res_map['frame']	# batch_l, source_l
		orig_l = self.shared.orig_seq_l

		batch_l, source_l, num_frame = log_frame.shape

		assert(self.opt.use_gold_predicate == 1)
		
		#loss = torch.zeros(1)
		#if self.opt.gpuid != -1:
		#	loss = to_device(loss, self.opt.gpuid)
		#
		#num_prop = 0
		#for i in range(batch_l):
		#	v_i = v_label[i, :v_l[i]]
		#	log_v_frame = log_frame[i, v_i]	# v_l[i], num_frame
		#	gold_v_frame = roleset_id[i, :v_l[i]]	# v_l[i],
		#	loss_i = -log_v_frame.gather(-1, gold_v_frame.unsqueeze(-1)).sum()
		#	loss = loss + loss_i
		#	num_prop += v_l[i]

		log_v_frame = batch_index1_select(log_frame, v_label, nul_idx=0)	# (batch_l, max_v_num, num_frame)
		loss_v_frame = -log_v_frame.gather(-1, roleset_id.unsqueeze(-1)).squeeze(-1)	# (batch_l, max_v_num)
		v_mask = (v_label != 0).float()
		loss = (loss_v_frame * v_mask).sum()

		num_prop = 0
		for i in range(batch_l):
			num_prop += v_l[i]
		self.num_prop += int(num_prop)

		# # average over number of predicates or num_ex
		normalizer = num_prop if self.opt.use_gold_predicate == 1 else sum([orig_l[i] for i in range(batch_l)])

		# stats
		v_frame_prime = log_v_frame.argmax(-1)
		for i in range(batch_l):
			num_correct = int((v_frame_prime[i, :v_l[i]] == roleset_id[i, :v_l[i]]).sum().item())
			self.num_correct += num_correct
		self.num_ex += batch_l
		frame_acc = float(self.num_correct) / self.num_prop

		#print('frame', loss / normalizer)
		return loss / normalizer, None


	# return a string of stats
	def print_cur_stats(self):
		if self.opt.use_gold_predicate == 1:
			frame_acc = float(self.num_correct) / self.num_prop
			stats = 'Frame acc {:.3f}'.format(frame_acc)
		else:
			assert(False)
		return stats

	# get training metric (scalar metric, extra metric)
	def get_epoch_metric(self):
		if self.opt.use_gold_predicate == 1:
			frame_acc = self.num_correct / self.num_prop
		else:
			assert(False)
		return frame_acc, [frame_acc]


	def begin_pass(self):
		self.num_correct = 0
		self.num_ex = 0
		self.num_prop = 0

	def end_pass(self):
		pass

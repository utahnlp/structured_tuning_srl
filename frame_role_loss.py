import sys
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from holder import *
from util import *

# loss on frame and role prediction
# Note that this loss only cares about the interaction of frame and role
class FrameRoleLoss(torch.nn.Module):
	def __init__(self, opt, shared):
		super(FrameRoleLoss, self).__init__()
		self.opt = opt
		self.shared = shared
		
		self.one = torch.ones(1).float()
		if opt.gpuid != -1:
			self.one = to_device(self.one, self.opt.gpuid)

		self.violation_cnt = 0
		self.num_prop = 0
		self.num_ex = 0

		self.use_gold_frame = hasattr(self.opt, 'use_gold_frame') and self.opt.use_gold_frame == 1

	def forward(self, log_pa, score, v_label, v_l, role_label, roleset_id, extra):
		assert(len(extra) != 0)
		log_frame = extra['frame']	# batch_l, source_l, num_frame
		frame_idx = self.shared.res_map['frame']	# batch_l, source_l
		frame_pool = self.shared.res_map['frame_pool']	# num_prop, num_frame, num_label
		orig_l = self.shared.orig_seq_l

		if self.opt.gpuid != -1:
			frame_pool = to_device(frame_pool, self.opt.gpuid)

		batch_l, source_l, num_frame = log_frame.shape

		assert(self.opt.use_gold_predicate == 1)
		loss = torch.zeros(1)
		if self.opt.gpuid != -1:
			loss = to_device(loss, self.opt.gpuid)

		#self._gold_roleset_id = roleset_id
		#self._gold_role_label = role_label

		num_prop = 0
		for i in range(batch_l):
			v_i = v_label[i, :v_l[i]]

			if self.opt.use_gold_predicate == 1:
				if not self.use_gold_frame:
					neg_role = (self.one - log_pa[i, v_i].exp()).clamp(min=1e-6).log().unsqueeze(1) # num_v, 1, source_l, num_label
					frame_pool_i = frame_pool[frame_idx[i, v_i]].unsqueeze(2)	# num_v, num_frame, 1, num_label
					frame_pred_i = log_frame[i, v_i]	# num_v, num_frame
					neg_role_mask = self.one - frame_pool_i.float()
	
					neg_role = neg_role_mask * neg_role	# num_v, num_frame, source_l, num_label
					loss_i = torch.relu(frame_pred_i - neg_role.min(-1)[0].min(-1)[0])	# num_v, num_frame
					loss_i = loss_i.sum()
				else:
					neg_role = (self.one - log_pa[i, v_i].exp()).clamp(min=1e-6).log().unsqueeze(1) # num_v, 1, source_l, num_label
					frame_pool_i = frame_pool[frame_idx[i, v_i]].unsqueeze(2)	# num_v, num_frame, 1, num_label
					neg_role_mask = self.one - frame_pool_i.float()

					# fake prediction by using gold
					roleset_idx_i = roleset_id[i, :v_l[i]].unsqueeze(-1)	# num_v, 1
					frame_pred_i = torch.zeros(v_l[i], num_frame).to(log_pa).scatter(-1, roleset_idx_i, 1.0)
					frame_pred_i = frame_pred_i.clamp(min=1e-6).log()
	
					neg_role = neg_role_mask * neg_role	# num_v, num_frame, source_l, num_label
					loss_i = torch.relu(frame_pred_i - neg_role.min(-1)[0].min(-1)[0])	# num_v, num_frame
					loss_i = loss_i.sum()
			else:
				assert(False)

			loss = loss + loss_i
			num_prop += v_l[i]

		# stats
		self.num_prop += int(num_prop)
		self.num_ex += batch_l

		# stats
		if hasattr(self.shared, 'viterbi_pred'):
			role_pred = self.shared.viterbi_pred
		else:
			role_pred = log_pa.argmax(-1)

		if not self.use_gold_frame:
			frame_pred = log_frame.argmax(-1)	# batch_l, source_l
			self._analyze(role_pred, frame_pred, frame_pool, frame_idx, v_label, v_l)
		else:
			frame_pred = torch.zeros(batch_l, orig_l.max()).long()
			frame_pred = to_device(frame_pred, self.opt.gpuid)
			for i in range(batch_l):
				for k in range(v_l[i]):
					frame_pred[i, v_label[i, k]] = roleset_id[i, k]
			self._analyze(role_pred, frame_pred, frame_pool, frame_idx, v_label, v_l)

		# # average over number of predicates or num_ex
		normalizer = float(num_prop) if self.opt.use_gold_predicate == 1 else sum([orig_l[i] for i in range(batch_l)])
		#print('framrrole', loss / normalizer)
		return loss / normalizer, None

	def _analyze(self, role_pred, frame_pred, frame_pool, frame_idx, v_label, v_l):
		batch_l, source_l = frame_pred.shape
		orig_l = self.shared.orig_seq_l
		num_label = self.opt.num_label

		if self.opt.use_gold_predicate == 1:
			for i in range(batch_l):
				v_i = v_label[i, :v_l[i]]
				frame_pred_i = frame_pred[i, v_i]	# num_v,
				frame_pool_i = frame_pool[frame_idx[i, v_i]]	# num_v, num_frame, num_label
				role_pred_i = role_pred[i, v_i, :orig_l[i]]		# num_v, orig_l

				#frame_pred_i = self._gold_roleset_id[i, :v_l[i]]
				#role_pred_i = self._gold_role_label[i, :v_l[i], :orig_l[i]]
				for k in range(v_l[i]):
					frame_pool_ik = frame_pool_i[k, frame_pred_i[k]]	# (num_label,)
					neg_role_ik = self.one - frame_pool_ik.float()		# (num_label,)
					role_pred_ik = role_pred_i[k]	# (orig_l,)
					invalid_role_pred = neg_role_ik.unsqueeze(0).expand(orig_l[i], num_label).gather(-1, role_pred_ik.unsqueeze(-1))	# (orig_l, 1)
					if invalid_role_pred.sum() > 0:
						self.violation_cnt += 1

		else:
			assert(False)


	# return a string of stats
	def print_cur_stats(self):
		rho = float(self.violation_cnt) / self.num_prop
		return "frame_role rho: {0:.3f}".format(rho)
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

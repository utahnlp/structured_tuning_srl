import sys
import torch
from torch import nn
from torch import cuda
from torch.autograd import Variable
from holder import *
import numpy as np
from optimizer import *
import time
from bert_encoder import *
from linear_classifier import *
from role_loss import *
from crf_loss import *
from frame_loss import *
from frame_role_loss import *
from unique_role_loss import *
from overlap_role_loss import *
from prep_modifier_loss import *
from continuous_role_loss import *

class Pipeline(torch.nn.Module):
	def __init__(self, opt, shared):
		super(Pipeline, self).__init__()

		self.shared = shared
		self.opt = opt
		self._loss_context = Holder()	# hidden container exclusively for loss calculation

		# pipeline stages
		if opt.enc == 'bert':
			self.encoder = BertEncoder(opt, shared)
		else:
			assert(False)

		if opt.cls == 'linear':
			self.classifier = LinearClassifier(opt, shared)
		else:
			assert(False)

		self.loss = []
		for l in self.opt.loss.split(','):
			if l == 'role':
				loss.append(RoleLoss(opt, shared))
			elif l == 'crf':
				self.loss.append(CRFLoss(opt, shared))
			elif l == 'frame':
				self.loss.append(FrameLoss(opt, shared))
			elif l == 'frame_role':
				self.loss.append(FrameRoleLoss(opt, shared))
			elif l == 'unique_role':
				self.loss.append(UniqueRoleLoss(opt, shared))
			elif l == 'overlap_role':
				self.loss.append(OverlapRoleLoss(opt, shared))
			elif l == 'prep_modifier':
				self.loss.append(PrepModifierLoss(opt, shared))
			elif l == 'continuous_role':
				self.loss.append(ContinuousRoleLoss(opt, shared))
			else:
				raise Exception("unrecognized loss {0}".format(l))
		self.loss = nn.ModuleList(self.loss)
		self.lambd = Variable(torch.Tensor([float(p) for p in opt.lambd.split(',')]), requires_grad=False)
		if self.opt.gpuid != -1:
			self.lambd = to_device(self.lambd, self.opt.gpuid)


	def init_weight(self):
		missed_names = []
		if self.opt.param_init_type == 'xavier_uniform':
			for n, p in self.named_parameters():
				if p.requires_grad and not hasattr(p, 'skip_init'):
					if 'weight' in n:
						print('initializing {}'.format(n))
						nn.init.xavier_uniform_(p)
					elif 'bias' in n:
						print('initializing {}'.format(n))
						nn.init.constant_(p, 0)
					else:
						missed_names.append(n)
				else:
					missed_names.append(n)
		elif self.opt.param_init_type == 'xavier_normal':
			for n, p in self.named_parameters():
				if p.requires_grad and not hasattr(p, 'skip_init'):
					if 'weight' in n:
						print('initializing {}'.format(n))
						nn.init.xavier_normal_(p)
					elif 'bias' in n:
						print('initializing {}'.format(n))
						nn.init.constant_(p, 0)
					else:
						missed_names.append(n)
				else:
					missed_names.append(n)
		elif self.opt.param_init_type == 'no':
			for n, p in self.named_parameters():
				missed_names.append(n)
		else:
			assert(False)

		if len(missed_names) != 0:
			print('uninitialized fields: {0}'.format(missed_names))


	def forward(self, tok_idx, skip_loss_forward=False):
		shared = self.shared

		# encoder
		enc = self.encoder(tok_idx)

		# classifier
		log_pa, score, extra = self.classifier(enc)

		assert(isinstance(self.loss[0], CRFLoss))

		if not skip_loss_forward:
			# always assume the first loss is crf loss which gives viterbi decoding
			loss_acc, pred = self.loss[0](log_pa, score, self._loss_context.v_label, self._loss_context.v_l, self._loss_context.role_label, self._loss_context.v_roleset_id, extra)
			#print('******* {0}'.format(loss_acc.data.item()))
			for k in range(1, len(self.loss)):
				l, _ = self.loss[k](log_pa, score, self._loss_context.v_label, self._loss_context.v_l, self._loss_context.role_label, self._loss_context.v_roleset_id, extra)
				#print(l.data.item())
				loss_acc = loss_acc + l * self.lambd[k]
		else:
			# skip loss forward pass, just do viterbi decoding
			loss_acc = None
			pred, _, _ = self.loss[0].decode(log_pa, score)

		return loss_acc, pred

	# update the contextual info of current batch
	def update_context(self, batch_ex_idx, batch_l, seq_l, orig_seq_l, sub2tok_idx, res_map=None):
		self.shared.batch_ex_idx = batch_ex_idx
		self.shared.batch_l = batch_l
		self.shared.seq_l = seq_l
		self.shared.orig_seq_l = orig_seq_l
		self.shared.sub2tok_idx = sub2tok_idx
		self.shared.res_map = res_map

	# loss context is only visible to the pipeline during loss computation (to avoid accidental contamination)
	# update the contextual info of current batch for loss calculation
	def update_loss_context(self, v_label, v_l, role_label, v_roleset_id):
		self._loss_context.v_label = v_label
		self._loss_context.v_l = v_l
		self._loss_context.role_label = role_label
		self._loss_context.v_roleset_id = v_roleset_id


	def begin_pass(self):
		self.encoder.begin_pass()
		self.classifier.begin_pass()
		for loss in self.loss:
			loss.begin_pass()

	def end_pass(self):
		self.encoder.end_pass()
		self.classifier.end_pass()
		for loss in self.loss:
			loss.end_pass()

	def print_cur_stats(self):
		log = []
		for l in self.loss:
			log.append(l.print_cur_stats())
		return ' '.join(log)

	def get_epoch_metric(self):
		head = None
		metrics = []
		for i, l in enumerate(self.loss):
			head_i, metrics_i = l.get_epoch_metric()
			# we only use the first loss metric to select model
			if i == 0:
				head = head_i
			metrics.extend(metrics_i)
		return head, metrics


	def distribute(self):
		modules = []
		modules.append(self.encoder)
		modules.append(self.classifier)
		for loss in self.loss:
			modules.append(loss)

		for m in modules:
			# This is no longer needed
			#if hasattr(m, 'fp16') and  m.fp16:
			#	m.half()

			if hasattr(m, 'customize_cuda_id'):
				print('pushing module to customized cuda id: {0}'.format(m.customize_cuda_id))
				m.cuda(m.customize_cuda_id)
			else:
				print('pushing module to default cuda id: {0}'.format(self.opt.gpuid))
				m.cuda(self.opt.gpuid)


	def get_param_dict(self):
		is_cuda = self.opt.gpuid != -1
		param_dict = {}
		skipped_fields = []
		for n, p in self.named_parameters():
			# save all parameters that do not have skip_save flag
			# 	unlearnable parameters will also be saved
			if not hasattr(p, 'skip_save') or p.skip_save == 0:
				param_dict[n] =  torch2np(p.data, is_cuda)
			else:
				skipped_fields.append(n)
		#print('skipped fields:', skipped_fields)
		return param_dict

	def set_param_dict(self, param_dict, verbose=True):
		skipped_fields = []
		rec_fields = []
		for n, p in self.named_parameters():
			if n in param_dict:
				rec_fields.append(n)
				# load everything we have
				if verbose:
					print('setting {0}'.format(n))
				p.data.copy_(torch.from_numpy(param_dict[n][:]))
			else:
				skipped_fields.append(n)
		print('skipped fileds: {0}'.format(skipped_fields))

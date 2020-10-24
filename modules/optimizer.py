import sys
import math
import torch
from torch import nn
from apex import amp
from transformers.optimization import *
from transformers import AdamW
from util.holder import *
from util.util import *


def warmup_cosine(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 0.5 * (1.0 + torch.cos(math.pi * x))

def warmup_constant(x, warmup=0.002):
    """ Linearly increases learning rate over `warmup`*`t_total` (as provided to BertAdam) training steps.
        Learning rate is 1. afterwards. """
    if x < warmup:
        return x/warmup
    return 1.0

def warmup_linear(x, warmup=0.002):
    """ Specifies a triangular learning rate schedule where peak is reached at `warmup`*`t_total`-th (as provided to BertAdam) training step.
        After `t_total`-th training step, learning rate is zero. """
    if x < warmup:
        return x/warmup
    return max((x-1.)/(warmup-1.), 0)

def warmup_linear_flat(x, warmup=0.002):
    """ Specifies a triangular learning rate schedule where peak is reached at `warmup`*`t_total`-th (as provided to BertAdam) training step.
        After `t_total`-th training step, learning rate is fixed. """
    if x < warmup:
        return x/warmup
    return 1.0


# the apex's adam for fp16 with huggingface AdamW
class AdamWFp16:
	def __init__(self, opt, shared):
		self.opt = opt
		self.shared = shared
		self.optim = None
		self.scheduler = None
		
	def build_optimizer(self, m, avg_batch_size=40):
		self.avg_batch_size = avg_batch_size
		no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
		named_params = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
		optimizer_grouped_parameters = [{'params': [p for n, p in named_params if not any(nd in n for nd in no_decay)], 'weight_decay': self.opt.weight_decay},
			{'params': [p for n, p in named_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

		adamw = AdamW(optimizer_grouped_parameters, lr=self.opt.learning_rate)
		m, self.optim = amp.initialize(m, adamw, opt_level='O1')
		return m

	def get_lr(self):
		if self.opt.warmup_perc <= 0:
			return self.opt.learning_rate
		acc_l = self.avg_batch_size if self.opt.acc_batch_size < 0 else self.opt.acc_batch_size
		normalizer = self.shared.num_train_ex / acc_l * self.opt.epochs
		return self.opt.learning_rate * warmup_linear_flat(self.shared.num_update / normalizer, self.opt.warmup_perc)

	def step(self, m):
		cur_lr = self.get_lr()
		for param_group in self.optim.param_groups:
			param_group['lr'] = cur_lr

		self.optim.step()

	# this interface is only for apex's optimizer
	def backward(self, m, loss):
		with amp.scale_loss(loss, self.optim) as scaled_loss:
			scaled_loss.backward()
		grad_norm2 = torch.nn.utils.clip_grad_norm_(amp.master_params(self.optim), self.opt.clip)

		return grad_norm2


def get_optimizer(opt, shared):
	optim = None
	if opt.optim == 'adamw_fp16':
		optim = AdamWFp16(opt, shared)
	else:
		print('unrecognized optim: {0}'.format(opt.optim))
		assert(False)
	return optim


def grad_sanity_check(optim, m, batch_size):
	optim.__SANITY_FLAG = False
	for n, p in m.named_parameters():
		if p.requires_grad:
			if p.grad is None:
				if optim.__SANITY_FLAG == False:
					print('{0} requires grad but has no grad, double check your graph'.format(n))
			else:
				if p.grad.is_sparse:
					print('sparse gradient found.')
					assert(False)
				p.grad.data.div_(batch_size)

	optim.__SANITY_FLAG = True



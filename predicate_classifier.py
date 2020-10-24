import sys
import torch
from torch import nn
from torch.autograd import Variable
from holder import *
from util import *

class PredicateClassifier(torch.nn.Module):
	def __init__(self, opt, shared):
		super(PredicateClassifier, self).__init__()
		self.opt = opt
		self.shared = shared

		# transformation to get phi_vs(x)
		self.f_v = nn.Sequential(
			nn.Linear(opt.hidden_size, opt.num_label))
			

	def _compact(self, enc):
		batch_l, source_l, hidden_size = enc.shape
		if self.opt.compact_mode == 'whole_word':
			enc = batch_index2_select(enc, self.shared.sub2tok_idx, nul_idx=-1)
			return enc.sum(2)	# (batch_l, seq_l, hidden_size)
		elif self.opt.compact_mode == 'first_subtok':
			enc = batch_index2_select(enc, self.shared.sub2tok_idx[:, :, :1], nul_idx=-1)
			return enc.squeeze(2)	# (batch_l, seq_l, hidden_size)
		else:
			raise Exception('unrecognized compact_mode: {}'.format(self.opt.compact_mode))


	def forward(self, enc):
		(batch_l, source_l, hidden_size) = enc.shape
		enc = self._compact(enc)

		v_score = self.f_v(enc.view(-1, hidden_size)).view(batch_l, source_l, self.opt.num_label)
		log_pv = nn.LogSoftmax(-1)(v_score)

		return log_pv, v_score, {}


	def begin_pass(self):
		pass

	def end_pass(self):
		pass


if __name__ == '__main__':
	pass





		
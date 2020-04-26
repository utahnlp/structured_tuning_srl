import sys
import torch
from torch import nn
from torch.autograd import Variable
from holder import *
from util import *

# linear classifier
class LinearClassifier(torch.nn.Module):
	def __init__(self, opt, shared):
		super(LinearClassifier, self).__init__()
		self.opt = opt
		self.shared = shared

		# transformation to get phi_vs(x)
		self.f_v = nn.Sequential(
			nn.Linear(opt.bert_size, opt.hidden_size))

		self.f_a = nn.Sequential(
			nn.Linear(opt.bert_size, opt.hidden_size))

		self.g_va = nn.Sequential(
			#nn.Dropout(opt.dropout),
			nn.Linear(opt.hidden_size*2, opt.hidden_size),
			nn.ReLU(),
			#nn.Dropout(opt.dropout),
			nn.Linear(opt.hidden_size, opt.hidden_size),
			nn.ReLU())

		self.label_layer = nn.Sequential(
			nn.Dropout(opt.dropout),
			nn.Linear(opt.hidden_size, opt.num_label))

		self.frame_layer = nn.Sequential(
			nn.Dropout(opt.dropout),
			nn.Linear(opt.hidden_size, opt.num_frame))
			

	def _compact(self, enc):
		batch_l, source_l, bert_size = enc.shape
		if self.opt.compact_mode == 'whole_word':
			enc = batch_index2_select(enc, self.shared.sub2tok_idx, nul_idx=-1)
			return enc.sum(2)	# (batch_l, seq_l, hidden_size)
		elif self.opt.compact_mode == 'first_subtok':
			enc = batch_index2_select(enc, self.shared.sub2tok_idx[:, :, :1], nul_idx=-1)
			return enc.squeeze(2)	# (batch_l, seq_l, hidden_size)
		else:
			raise Exception('unrecognized compact_mode: {}'.format(self.opt.compact_mode))


	def forward(self, enc):
		(batch_l, source_l, bert_size) = enc.shape
		enc = self._compact(enc)

		v_enc = self.f_v(enc.view(-1, bert_size)).view(batch_l, source_l, 1, self.opt.hidden_size)
		a_enc = self.f_a(enc.view(-1, bert_size)).view(batch_l, 1, source_l, self.opt.hidden_size)
		# forming a large tensor
		va_enc = torch.cat([
			v_enc.expand(batch_l, source_l, source_l, self.opt.hidden_size),
			a_enc.expand(batch_l, source_l, source_l, self.opt.hidden_size)], dim=-1)
		va_enc = self.g_va(va_enc.view(-1, self.opt.hidden_size*2))
		va_enc = va_enc.view(batch_l, source_l, source_l, self.opt.hidden_size)

		# compute score of ijl
		a_score = self.label_layer(va_enc.view(-1, self.opt.hidden_size)).view(batch_l, source_l, source_l, self.opt.num_label)
		log_pa = nn.LogSoftmax(-1)(a_score)

		extra = {}
		if hasattr(self, 'frame_layer'):
			frame_score = self.frame_layer(v_enc.view(-1, self.opt.hidden_size)).view(batch_l, source_l, -1)
			log_frame = nn.LogSoftmax(-1)(frame_score)
			extra['frame'] = log_frame

		return log_pa, a_score, extra


	def begin_pass(self):
		pass

	def end_pass(self):
		pass


if __name__ == '__main__':
	pass





		
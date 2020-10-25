import sys
import torch
from torch import nn
from torch.autograd import Variable
from util.holder import *
from util.util import *
from .crf import *
import time

# CRF loss function incl. decoding
class CRFLoss(torch.nn.Module):
	def __init__(self, opt, shared):
		super(CRFLoss, self).__init__()
		self.opt = opt
		self.shared = shared

		self.labels = np.asarray(self.opt.labels)

		constraints = allowed_transitions("BIO", self.opt.label_map_inv)

		self.crf = ConditionalRandomField(num_tags=opt.num_label, constraints=constraints, gpuid=opt.gpuid)

		self.quick_acc_sum = 0.0
		self.num_ex = 0
		self.inconsistent_bio_cnt = 0
		self.gold_log = []
		self.pred_log = []

	def decode(self, log_pa, score, v_label=None, v_l=None):
		batch_l, source_l, _, _ = score.shape
		orig_l = self.shared.orig_seq_l
		max_orig_l = orig_l.max()
		bv_idx = int(np.where(self.labels == 'B-V')[0][0])
		max_num_v = 30

		a_score = []
		a_mask = []

		if v_label is None:
			v_label = to_device(torch.zeros(batch_l, max_orig_l).long(), self.opt.gpuid)
			v_l = to_device(torch.zeros(batch_l).long(), self.opt.gpuid)

			# use heuristic to get predicates
			for i in range(batch_l):
				max_v_idx = (score[i].argmax(-1) == bv_idx).diagonal().nonzero().view(-1)
				# if no predicate candidate found, just take the one with the max score on B-V
				if max_v_idx.numel() == 0:
					max_v_idx = score[i, :, :, bv_idx].max(-1)[0].argmax(-1).view(1)
				max_v_idx = max_v_idx[:max_num_v]
				v_l[i] = max_v_idx.shape[0]
				v_label[i, :v_l[i]] = max_v_idx
		else:
			v_label = to_device(v_label, self.opt.gpuid)
			v_l = to_device(v_l, self.opt.gpuid)

		# pack everything into (batch_l*acc_orig_l, max_orig_l, ...)
		for i in range(batch_l):

			a_mask_i = torch.zeros(v_l[i], max_orig_l).byte()
			a_mask_i[:, :orig_l[i]] = True
			a_mask.append(a_mask_i)
			
			a_score_i = score[i].index_select(0, v_label[i, :v_l[i]])[:, :max_orig_l]
			a_score.append(a_score_i)

		a_score = to_device(torch.cat(a_score, dim=0), self.opt.gpuid)
		a_mask = to_device(torch.cat(a_mask, dim=0), self.opt.gpuid)

		decoded = self.crf.viterbi_tags(a_score, a_mask)

		# unpack pred_idx to (batch_l, max_orig_l, max_orig_l, ...)
		pred_idx = torch.zeros(batch_l, max_orig_l, max_orig_l).long()
		row_idx = 0
		for i in range(batch_l):
			for k in range(v_l[i]):
				pred_idx[i, v_label[i, k], :orig_l[i]] = torch.Tensor(decoded[row_idx][0]).long()
				row_idx += 1
		assert(row_idx == len(decoded))
		pred_idx = to_device(pred_idx, self.opt.gpuid)

		return pred_idx, {'v_label': v_label, 'v_l': v_l}


	def forward(self, log_pa, score, v_label, v_l, role_label, roleset_id, extra={}):
		batch_l, source_l, _, _ = score.shape
		orig_l = self.shared.orig_seq_l
		max_orig_l = orig_l.max()

		if self.shared.is_train:
			a_score = []
			a_gold = []
			a_mask = []
	
			if self.opt.use_gold_predicate == 1:
				# pack everything into (batch_l*acc_v_l, max_orig_l, ...)
				for i in range(batch_l):
					v_i = v_label[i, :v_l[i]]
					a_gold_i = torch.zeros(v_l[i], max_orig_l).long()	# O has idx 0
					a_gold_i[:, :orig_l[i]] = role_label[i, :v_l[i], :orig_l[i]]	# (num_v, orig_l)
					a_gold.append(a_gold_i)
	
					a_mask_i = torch.zeros(v_l[i], max_orig_l).byte()
					a_mask_i[:, :orig_l[i]] = True
					a_mask.append(a_mask_i)
	
					a_score_i = score[i].index_select(0, v_label[i, :v_l[i]])[:, :max_orig_l]
					a_score.append(a_score_i)
			else:
				# pack everything into (batch_l*acc_orig_l, max_orig_l, ...)
				for i in range(batch_l):
					v_i = v_label[i, :v_l[i]]
					role_i = role_label[i, :v_l[i], :orig_l[i]]	# (num_v, orig_l)
					a_gold_i = torch.zeros(orig_l[i], max_orig_l).long()	# O has idx 0
					for k, role_k in enumerate(role_i):
						a_gold_i[v_i[k], :orig_l[i]] = role_k
					a_gold.append(a_gold_i)
		
					a_mask_i = torch.zeros(orig_l[i], max_orig_l).byte()
					a_mask_i[:, :orig_l[i]] = True
					a_mask.append(a_mask_i)
	
					a_score.append(score[i, :orig_l[i], :max_orig_l])
	
			a_gold = to_device(torch.cat(a_gold, dim=0), self.opt.gpuid)
			a_score = to_device(torch.cat(a_score, dim=0), self.opt.gpuid)
			a_mask = to_device(torch.cat(a_mask, dim=0), self.opt.gpuid)

			# pred_idx (batch_l, max_orig_l, max_orig_l, ...)
			pred_idx = log_pa[:, :max_orig_l, :max_orig_l].argmax(-1)

			loss = self.crf(a_score, a_gold, a_mask)

			# analyze
			pred_v_roles = batch_index1_select(pred_idx, v_label, nul_idx=0)	# batch_l, max_v_num, max_orig_l
			pred_v_roles = pred_v_roles[:, :role_label.shape[1], :]	# "only" take the gold predicates
			batch_acc_sum = self._count_quick_acc(pred_v_roles, role_label) * batch_l
			self.quick_acc_sum += batch_acc_sum
		else:
			# during evaluation, no need to compute the loss, just =0
			loss = to_device(torch.zeros(1), self.opt.gpuid)

			pred_idx, extra_pred = self.decode(log_pa, score)
			v_label = extra_pred['v_label']
			v_l = extra_pred['v_l']

		self.num_ex += batch_l
		self.shared.viterbi_pred = pred_idx

		if not self.shared.is_train:
			self.analyze(pred_idx, v_label, v_l, role_label)

		# # average over number of predicates or num_ex
		num_prop = sum([v_l[i] for i in range(batch_l)])
		normalizer = float(num_prop) if self.opt.use_gold_predicate == 1 else sum([orig_l[i] for i in range(batch_l)])
		#print('crf', loss / normalizer)
		return loss / normalizer, pred_idx


	# return rough estimate of accuracy that counts only non-O elements (in both pred and gold)
	def _count_quick_acc(self, pred_idx, gold_idx):
		pred_mask = pred_idx != 0
		gold_mask = gold_idx != 0
		non_o_mask = ((pred_mask + gold_mask) > 0).int()
		overlap = (pred_idx == gold_idx).int() * non_o_mask
		if non_o_mask.sum() != 0:
			return float(overlap.sum().item()) / non_o_mask.sum().item()
		else:
			return 1.0


	def analyze(self, pred_idx, v_label, v_l, role_label):
		batch_l = self.shared.batch_l
		orig_l = self.shared.orig_seq_l
		bv_idx = int(np.where(self.labels == 'B-V')[0][0])

		for i in range(batch_l):
			orig_l_i = orig_l[i].item()	# convert to scalar
			v_i = v_label[i, :v_l[i]]
			role_i = role_label[i, :v_l[i], :orig_l_i]	# (num_v, orig_l)

			a_gold_i = torch.zeros(orig_l_i, orig_l_i).long()	# O has idx 0
			for k, role_k in enumerate(role_i):
				a_gold_i[v_i[k]] = role_k

			a_pred_i = pred_idx[i, :orig_l_i, :orig_l_i]
			# if using gold predicate during evaluation, wipe out non-predicate predictions
			if self.opt.use_gold_predicate == 1:
				a_pred_i_new = torch.zeros(orig_l_i, orig_l_i).long()	# O has idx 0
				for k, _ in enumerate(role_i):
					a_pred_i_new[v_i[k]] = a_pred_i[v_i[k]]
					a_pred_i_new[v_i[k], v_i[k]] = bv_idx	# force it to be B-V, this could cause inconsistent BIO labels in some rare cases
				a_pred_i = a_pred_i_new

			# do analysis without cls and sep
			orig_tok_grouped = self.shared.res_map['orig_tok_grouped'][i][1:-1]
			self.gold_log.append(self.compose_log(orig_tok_grouped, a_gold_i[1:-1, 1:-1]))
			self.pred_log.append(self.compose_log(orig_tok_grouped, a_pred_i[1:-1, 1:-1]))


	# return a string of stats
	def print_cur_stats(self):
		stats = 'Quick acc {:.3f}'.format(self.quick_acc_sum / self.num_ex)
		return stats

	# get training metric (scalar metric, extra metric)
	def get_epoch_metric(self):
		if self.inconsistent_bio_cnt != 0:
			print('warning: inconsistent_bio_cnt: ', self.inconsistent_bio_cnt)
		if self.shared.is_train:
			quick_acc = self.quick_acc_sum / self.num_ex
			return quick_acc, [quick_acc] 	# and any other scalar metrics	
		else:
			print('writing gold to {}'.format(self.opt.conll_output + '.gold.txt'))
			with open(self.opt.conll_output + '.gold.txt', 'w') as f:
				for ex in self.gold_log:
					f.write(ex + '\n')
	
			print('writing pred to {}'.format(self.opt.conll_output + '.pred.txt'))
			with open(self.opt.conll_output + '.pred.txt', 'w') as f:
				for ex in self.pred_log:
					f.write(ex + '\n')

			f1 = system_call_eval(self.opt.conll_output + '.gold.txt', self.opt.conll_output + '.pred.txt')
			return f1, [f1]

	# compose log for one example
	#	role_labels of shape (seq_l, seq_l)
	def compose_log(self, orig_toks, role_labels, transpose=True):
		role_labels = role_labels.cpu().numpy()
		seq_l = role_labels.shape[0]
#
		header = ['-' for _ in range(seq_l)]
		role_lines = []
		for i, row in enumerate(role_labels):
			roles = self.labels[row].tolist()
			roles = roles + ['O']	# TODO, the convert_role_labels prefers the last label to be O, so bit hacky here
			if 'B-V' in roles:
				v_idx = i if roles[i] == 'B-V' else roles.index('B-V')
				header[v_idx] = orig_toks[v_idx]
				roles, error_cnt = convert_role_labels(roles)
				role_lines.append(roles[:-1])
				self.inconsistent_bio_cnt += error_cnt
#
		log = [header] + role_lines
		log = np.asarray(log)
		# do a transpose
		if transpose:
			log = log.transpose((1, 0))
		
		rs = []
		for row in log:
			rs.append(' '.join(row))
		return '\n'.join(rs) + '\n'


	def begin_pass(self):
		# clear stats
		self.quick_acc_sum = 0
		self.num_ex = 0
		self.gold_log = []
		self.pred_log = []
		self.inconsistent_bio_cnt = 0

	def end_pass(self):
		pass

if __name__ == '__main__':
	pass





		
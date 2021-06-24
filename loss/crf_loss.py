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

		self.labels = self.opt.labels
		self.label_groups = []
		self.label_group_map = {}
		self.label_map_inv = self.opt.label_map_inv

		for i, l in enumerate(self.labels):
			group = l[2:] if l != 'O' else l # take off B- and I-
			if group.startswith('C-') or group.startswith('R-'):
				group = group[2:]
			if group not in self.label_groups:
				self.label_groups.append(group)
			self.label_group_map[i] = self.label_groups.index(group)

		self.labels = np.asarray(self.labels)

		constraints = allowed_transitions("BIO", self.label_map_inv)

		self.crf = ConditionalRandomField(num_tags=opt.num_label, constraints=constraints, gpuid=opt.gpuid)

		self.quick_acc_sum = 0.0
		self.num_ex = 0
		self.inconsistent_bio_cnt = 0
		self.gold_log = []	# log used for srl eval
		self.pred_log = []
		self.pretty_gold = []	# log used for pretty print
		self.pretty_pred = []
		self.orig_toks = []
		self.conf_map = torch.zeros(len(self.opt.labels), len(self.opt.labels))

		self.log_types = []
		if hasattr(self.opt, 'logs'):
			self.log_types = self.opt.logs.strip().split(',')

	# the decode function is for the demo where gold predicate might not present
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
				a_mask.append(mask_i)

				a_score.append(score[i, :orig_l[i], :max_orig_l])

		a_gold = to_device(torch.cat(a_gold, dim=0), self.opt.gpuid)
		a_score = to_device(torch.cat(a_score, dim=0), self.opt.gpuid)
		a_mask = to_device(torch.cat(a_mask, dim=0), self.opt.gpuid)

		if self.shared.is_train:
			# pred_idx (batch_l, max_orig_l, max_orig_l, ...)
			pred_idx = log_pa[:, :max_orig_l, :max_orig_l].argmax(-1).cpu()
			loss = self.crf(a_score, a_gold, a_mask)
		else:
			# in validation mode, no need to count the loss here
			loss = torch.zeros(1)
			if self.opt.gpuid != -1:
				loss = to_device(loss, self.opt.gpuid)

			decoded = self.crf.viterbi_tags(a_score, a_mask)
			# unpack pred_idx to (batch_l, max_orig_l, max_orig_l, ...)
			pred_idx = torch.zeros(batch_l, max_orig_l, max_orig_l).long()

			if self.opt.use_gold_predicate == 1:
				row_idx = 0
				for i in range(batch_l):
					for k in range(v_l[i]):
						pred_idx[i, v_label[i, k], :orig_l[i]] = torch.Tensor(decoded[row_idx][0]).long()
						row_idx += 1
				assert(row_idx == len(decoded))
			else:
				acc_l = 0
				for i in range(batch_l):
					pred_idx[i, :orig_l[i], :orig_l[i]] = torch.Tensor([p[0] for p in decoded[acc_l:acc_l+orig_l[i]]]).long()
					acc_l += orig_l[i]
		
		pred_idx = to_device(pred_idx, self.opt.gpuid)

		# analyze
		pred_v_roles = batch_index1_select(pred_idx, v_label, nul_idx=0)	# batch_l, max_v_num, max_orig_l
		pred_v_roles = pred_v_roles[:, :role_label.shape[1], :]	# "only" take the gold predicates
		batch_acc_sum = self._count_quick_acc(pred_v_roles, role_label) * batch_l
		self.quick_acc_sum += batch_acc_sum
		self.num_ex += batch_l
		self.shared.viterbi_pred = pred_idx

		if not self.shared.is_train:
			self.analyze(pred_idx, v_label, v_l, role_label, roleset_id)

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


	def analyze(self, pred_idx, v_label, v_l, role_label, roleset_id):
		batch_l = self.shared.batch_l
		orig_l = self.shared.orig_seq_l
		bv_idx = int(np.where(self.labels == 'B-V')[0][0])
		frame_idx = self.shared.res_map['frame']	# batch_l, source_l
		frame_pool = self.shared.res_map['frame_pool']	# num_prop, num_frame, num_label

		for i in range(batch_l):
			orig_l_i = orig_l[i].item()	# convert to scalar
			v_i = v_label[i, :v_l[i]]
			role_i = role_label[i, :v_l[i], :orig_l_i]	# (num_v, orig_l)
			gold_frame_pool_i = frame_pool[frame_idx[i, v_i]]

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

			if 'pretty' in self.log_types:
				self.orig_toks.append(orig_tok_grouped)

				gold_log = self.compose_log(orig_tok_grouped, a_gold_i[1:-1, 1:-1]).split('\n')
				gold_log = gold_log[:-1]	# rip the last newline
				assert(len(gold_log) == len(orig_tok_grouped))
				gold_log = [tok + ' ' + row for tok, row in zip(orig_tok_grouped, gold_log)]
				gold_frame_log = self.compose_frame_log(gold_frame_pool_i, self.shared.res_map['orig_tok_grouped'][i], v_i, a_gold_i[v_i], roleset_id[i, :v_l[i]])
				gold_log = gold_log + ['-------- FRAME --------'] + gold_frame_log
				self.pretty_gold.append('\n'.join(gold_log) + '\n')

				pred_log = self.compose_log(orig_tok_grouped, a_pred_i[1:-1, 1:-1]).split('\n')
				pred_log = pred_log[:-1]	# rip the last newline
				assert(len(pred_log) == len(orig_tok_grouped))
				pred_log = [tok + ' ' + row for tok, row in zip(orig_tok_grouped, pred_log)]
				self.pretty_pred.append('\n'.join(pred_log) + '\n')

			if 'confusion' in self.log_types and self.opt.use_gold_predicate == 1:
				for k, _ in enumerate(role_i):
					g = a_gold_i[v_i[k]]
					p = a_pred_i[v_i[k]]
					for p_j, g_j in zip(p, g):
						p_j, g_j = self.label_group_map[int(p_j)], self.label_group_map[int(g_j)]
						self.conf_map[g_j, p_j] += 1


	def compose_frame_log(self, frame_pool, orig_tok_grouped, v_label, role_label, roleset_id):
		def merge_roles(roles):
			unique = roles
			unique = set([r[2:] if r.startswith('B-') or r.startswith('I-') else r for r in unique])
			unique = set([r[2:] if r.startswith('C-') or r.startswith('R-') else r for r in unique])
			return sorted(list(unique))

		log = []
		for k in range(len(v_label)):
			valid_roles = frame_pool[k, roleset_id[k]]
			roleset_name = self.opt.roleset_map_inv[int(roleset_id[k].item())]
			roles = torch.zeros(valid_roles.shape).scatter(0, role_label[k].unique(), 1)
			has_violation = ((valid_roles - roles) < 0).any()
			has_violation = 'HAS_VIOLATION' if has_violation else 'GOOD'

			if (valid_roles[1:] == 0).all():
				log.append('DISABLED_ROLE {0}: {1}|ALL {2}'.format(orig_tok_grouped[v_label[k]], roleset_name, has_violation))
			elif (valid_roles[1:] == 1).all():
				log.append('DISABLED_ROLE {0}: {1}|NONE {2}'.format(orig_tok_grouped[v_label[k]], roleset_name, has_violation))	
			else:
				valid_role_idx = [j for j in range(len(valid_roles)) if valid_roles[j] == 0]
				log.append('DISABLED_ROLE {0}: {1}|{2} {3}'.format(orig_tok_grouped[v_label[k]], roleset_name, ','.join(merge_roles([self.opt.label_map_inv[j] for j in valid_role_idx])), has_violation))
		return log


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
		role_labels = role_labels.numpy()
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
		self.pretty_gold = []
		self.pretty_pred = []
		self.orig_toks = []
		self.inconsistent_bio_cnt = 0
		self.conf_map = torch.zeros(len(self.label_groups), len(self.label_groups))
		self.frame_log = []

	def end_pass(self):
		if 'pretty' in self.log_types:
			print('writing pretty gold to {}'.format(self.opt.conll_output + '.pretty_gold.txt'))
			with open(self.opt.conll_output + '.pretty_gold.txt', 'w') as f:
				for toks, ex in zip(self.orig_toks, self.pretty_gold):
					#f.write(' '.join(toks)+'\n')
					f.write(ex + '\n')
			print('writing pretty pred to {}'.format(self.opt.conll_output + '.pretty_pred.txt'))
			with open(self.opt.conll_output + '.pretty_pred.txt', 'w') as f:
				for toks, ex in zip(self.orig_toks, self.pretty_pred):
					#f.write(' '.join(toks)+'\n')
					f.write(ex + '\n')

		if 'confusion' in self.log_types and self.opt.use_gold_predicate == 1:

			label_cnt = self.conf_map.sum(-1).tolist()
			sorted_labels = sorted([(i,j) for i, j in zip(self.label_groups, label_cnt)], key=lambda x:x[1], reverse=True)
			sorted_labels = [i for i, _ in sorted_labels]
			sorted_label_idx = [self.label_groups.index(l) for l in sorted_labels]
			sorted_conf_map = [[row[i] for i in sorted_label_idx] for row in self.conf_map[sorted_label_idx]]

			print('writing confusion matrix to {}'.format(self.opt.conll_output + '.confusion.txt'))
			with open(self.opt.conll_output + '.confusion.txt', 'w') as f:
				f.write('gold\\pred\t' + '\t'.join(sorted_labels) + '\n')
				for i, row in enumerate(sorted_conf_map):
					f.write(sorted_labels[i] + '\t' + '\t'.join([str(int(p)) for p in row]) + '\n')

if __name__ == '__main__':
	pass




		
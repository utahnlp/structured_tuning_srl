import io
import h5py
import torch
from torch import nn
from torch import cuda
import numpy as np
import ujson
from util import *

class Data():
	def __init__(self, opt, data_file, res_files=None, triple_mode=False):
		self.opt = opt
		self.data_name = data_file

		print('loading data from {0}'.format(data_file))
		f = h5py.File(data_file, 'r')
		self.tok_idx = f['tok_idx'][:]	# indices to glove tokens
		self.seq_l = f['seq_l'][:].astype(np.int32)
		self.orig_seq_l = f['orig_seq_l'][:].astype(np.int32)
		self.sub2tok_idx = f['sub2tok_idx'][:].astype(np.int32)
		self.v_idx = f['v_idx'][:].astype(np.int32)
		self.role_label = f['role_label'][:].astype(np.int32)
		self.v_l = f['v_l'][:].astype(np.int32)
		self.v_roleset_id = f['v_roleset_id'][:].astype(np.int32)
		self.batch_l = f['batch_l'][:].astype(np.int32)
		self.batch_idx = f['batch_idx'][:].astype(np.int32)
		self.ex_idx = f['ex_idx'][:].astype(np.int32)
		self.length = self.batch_l.shape[0]

		self.tok_idx = torch.from_numpy(self.tok_idx)
		self.seq_l = torch.from_numpy(self.seq_l)
		self.orig_seq_l = torch.from_numpy(self.orig_seq_l)
		self.sub2tok_idx = torch.from_numpy(self.sub2tok_idx)
		self.v_idx = torch.from_numpy(self.v_idx)
		self.role_label = torch.from_numpy(self.role_label)
		self.v_l = torch.from_numpy(self.v_l)
		self.v_roleset_id = torch.from_numpy(self.v_roleset_id)

		self.batches = []
		for i in range(self.length):
			start = self.batch_idx[i]
			end = start + self.batch_l[i]

			# get example token indices
			tok_idx_i = self.tok_idx[start:end, 0:self.seq_l[i]]
			sub2tok_idx_i = self.sub2tok_idx[start:end, 0:self.seq_l[i]]
			v_idx_i = self.v_idx[start:end]
			v_l_i = self.v_l[start:end]
			v_roleset_id_i = self.v_roleset_id[start:end]
			orig_seq_l_i = self.orig_seq_l[start:end]
			role_label_i = self.role_label[start:end, 0:v_l_i.max(), 0:orig_seq_l_i.max()]

			# sanity check
			assert(self.tok_idx[start:end, self.seq_l[i]:].sum() == 0)
			for k in range(start, end):
				assert((self.sub2tok_idx[k, self.orig_seq_l[k]:] != -1).sum() == 0)
				assert(self.v_idx[k, self.v_l[k]:].sum() == 0)
				assert((self.v_idx[k, :self.v_l[k]] == 0).sum() == 0)	# there is no zero (0-th is <BOS>)

			self.batches.append((tok_idx_i, self.seq_l[i], orig_seq_l_i, sub2tok_idx_i, v_idx_i, v_l_i, role_label_i, v_roleset_id_i))

		# count examples
		self.num_ex = 0
		for i in range(self.length):
			self.num_ex += self.batch_l[i]

		# load resource files
		self.res_names = []
		if res_files is not None:
			for f in res_files:
				print('loading resource from {0}'.format(f))
				if f.endswith('txt'):
					res_names = self.__load_txt(f)

				elif f.endswith('json'):
					res_names = self.__load_json_res(f)

				elif f.endswith('frame_pool.hdf5'):
					file = h5py.File(f, 'r')
					self.frame_pool = torch.from_numpy(file['frame_pool'][:])
					res_names = ['frame_pool']

				elif f.endswith('hdf5'):
					res_names = self.__load_hdf5(f)

				else:
					assert(False)
				self.res_names.extend(res_names)


	def subsample(self, ratio, minimal_num=0):
		target_num_ex = int(float(self.num_ex) * ratio)
		target_num_ex = max(target_num_ex, minimal_num)
		sub_idx = torch.LongTensor(range(self.size()))
		sub_num_ex = 0

		if ratio != 1.0:
			rand_idx = torch.randperm(self.size())
			i = 0
			while sub_num_ex < target_num_ex and i < self.batch_l.shape[0]:
				sub_num_ex += self.batch_l[rand_idx[i]]
				i += 1
			sub_idx = rand_idx[:i]

		else:
			sub_num_ex = self.batch_l.sum()

		return sub_idx, sub_num_ex

	def split(self, sub_idx, ratio):
		num_ex = sum([self.batch_l[i] for i in sub_idx])
		target_num_ex = int(float(num_ex) * ratio)

		cur_num_ex = 0
		cur_pos = 0
		for i in range(len(sub_idx)):
			cur_pos = i
			cur_num_ex += self.batch_l[sub_idx[i]]
			if cur_num_ex >= target_num_ex:
				break

		return sub_idx[:cur_pos+1], sub_idx[cur_pos+1:], cur_num_ex, num_ex - cur_num_ex


	def __load_txt(self, path):
		lines = []
		# read file in unicode mode!!!
		with io.open(path, 'r+', encoding="utf-8") as f:
			for l in f:
				lines.append(l.rstrip())
		# the second last extension is the res name
		res_name = path.split('.')[-2]
		res_data = lines[:]

		# some customized parsing
		parsed = []
		if res_name == 'orig_tok_grouped':
			for l in res_data:
				parsed.append(l.rstrip().split(' '))
		else:
			parsed = res_data

		setattr(self, res_name, parsed)
		return [res_name]


	def __load_hdf5(self, path):
		res_name = path.split('.')[-2]
		file = h5py.File(path, 'r')
		setattr(self, res_name, torch.from_numpy(file[res_name][:]))
		return [res_name]


	def __load_json_res(self, path):
		f_str = None
		with open(path, 'r') as f:
			f_str = f.read()
		j_obj = ujson.loads(f_str)

		# get key name of the file
		assert(len(j_obj) == 2)
		res_type = next(iter(j_obj))

		res_name = None
		if j_obj[res_type] == 'map':
			res_name = self.__load_json_map(path)
		elif j_obj[res_type] == 'list':
			res_name = self.__load_json_list(path)
		else:
			assert(False)

		return [res_name]

	
	def __load_json_map(self, path):
		f_str = None
		with open(path, 'r') as f:
			f_str = f.read()
		j_obj = ujson.loads(f_str)

		assert(len(j_obj) == 2)

		res_name = None
		for k, v in j_obj.items():
			if k != 'type':
				res_name = k

		# optimize indices
		res = {}
		for k, v in j_obj[res_name].items():
			lut = {}
			for i, j in v.items():
				if i == res_name:
					lut[res_name] = [int(l) for l in j]
				else:
					lut[int(i)] = ([l for l in j[0]], [l for l in j[1]])

			res[int(k)] = lut
		
		setattr(self, res_name, res)
		return res_name


	def __load_json_list(self, path):
		f_str = None
		with open(path, 'r') as f:
			f_str = f.read()
		j_obj = ujson.loads(f_str)

		assert(len(j_obj) == 2)
		
		res_name = None
		for k, v in j_obj.items():
			if k != 'type':
				res_name = k

		# optimize indices
		res = {}
		for k, v in j_obj[res_name].items():
			p = v['p']
			h = v['h']

			# for token indices, shift by 1 to incorporate the nul-token at the beginning
			res[int(k)] = ([l for l in p], [l for l in h])
		
		setattr(self, res_name, res)
		return res_name


	def size(self):
		return self.length


	def __getitem__(self, idx):
		(tok_idx, seq_l, orig_seq_l, sub2tok_idx, v_idx, v_l, role_label, v_roleset_id) = self.batches[idx]

		# convert all indices to long format
		tok_idx = tok_idx.long()
		sub2tok_idx = sub2tok_idx.long()
		v_idx = v_idx.long()
		role_label = role_label.long()
		v_roleset_id = v_roleset_id.long()

		# transfer to gpu if needed
		if self.opt.gpuid != -1:
			tok_idx = tok_idx.cuda(self.opt.gpuid)
			sub2tok_idx = sub2tok_idx.cuda(self.opt.gpuid)
			v_idx = v_idx.cuda(self.opt.gpuid)
			role_label = role_label.cuda(self.opt.gpuid)
			v_roleset_id = v_roleset_id.cuda(self.opt.gpuid)

		# get batch ex indices
		batch_ex_idx = [self.ex_idx[i] for i in range(self.batch_idx[idx], self.batch_idx[idx] + self.batch_l[idx])]
		res_map = self.__get_res(idx)

		return (self.data_name, tok_idx, batch_ex_idx, self.batch_l[idx], seq_l, orig_seq_l, sub2tok_idx, v_idx, v_l, role_label, v_roleset_id, res_map)
		

	def __get_res(self, idx):
		# if there is no resource presents, return None
		if len(self.res_names) == 0:
			return None

		batch_ex_idx = [self.ex_idx[i] for i in range(self.batch_idx[idx], self.batch_idx[idx] + self.batch_l[idx])]

		all_res = {}
		for res_n in self.res_names:
			res = getattr(self, res_n)
			if res_n == 'frame_pool':	# frame_pool is invariant across examples
				all_res[res_n] = self.frame_pool
			elif type(res) == list:
				batch_res = [res[ex_id] for ex_id in batch_ex_idx]
				all_res[res_n] = batch_res
			else:
				all_res[res_n] = res[batch_ex_idx]

		return all_res


	# something at the beginning of each pass of training/eval
	#	e.g. setup preloading
	def begin_pass(self):
		pass


	def end_pass(self):
		pass

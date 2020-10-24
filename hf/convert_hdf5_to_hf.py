import sys
import argparse
import h5py
import numpy as np
import torch
from transformers import *
from util.holder import *
from util.util import *
from .roberta_for_srl import *
from modules import pipeline
import logging

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--load_file', help="Path to where HDF5 model to be loaded.", default="")
parser.add_argument('--label_dict', help="The path to label dictionary", default = "./data/srl/conll2012.label.dict")
# bert specs
parser.add_argument('--bert_type', help="The type of bert encoder from huggingface, eg. roberta-base",default = "roberta-base")
parser.add_argument('--compact_mode', help="How word pieces be mapped to word level label", default='whole_word')
## pipeline stages
parser.add_argument('--enc', help="The type of encoder, bert", default='bert')
parser.add_argument('--cls', help="The type of classifier, linear", default='linear')
#
parser.add_argument('--output', help="Path to output HuggingFace(HF) format", default='/models/hf')
parser.add_argument('--num_frame', help="The number of frame for each proposition", type=int, default=38)
parser.add_argument('--max_num_subtok', help="Maximal number subtokens in a word", type=int, default=8)


def main(args):
	opt = parser.parse_args(args)
	opt = complete_opt(opt)
	opt.gpuid = -1
	opt.dropout = 0
	opt.loss = 'crf'
	opt.lambd = "1.0"
	shared = Holder()

	# load model
	m = pipeline.Pipeline(opt, shared)
	print('loading pretrained model from {0}...'.format(opt.load_file))
	param_dict = load_param_dict('{0}.hdf5'.format(opt.load_file))
	m.set_param_dict(param_dict)

	mlm = AutoModel.from_pretrained(opt.bert_type)
	tokenizer = AutoTokenizer.from_pretrained(opt.bert_type)

	config = mlm.config
	config.gpuid = -1
	config.dropout = opt.dropout
	config.compact_mode = opt.compact_mode
	config.num_frame = opt.num_frame
	config.max_num_subtok = opt.max_num_subtok
	config.bert_type = opt.bert_type
	config.labels, config.label_map_inv = opt.labels, opt.label_map_inv
	config.lambd = opt.lambd
	config.num_label=len(config.labels)

	if 'roberta' in opt.bert_type:
		m_hf = RobertaForSRL(config, shared=Holder())
		m_hf.config = config
		# move parameters
		m_hf.roberta = m.encoder.bert
		m_hf.classifier = m.classifier
		m_hf.crf_loss = m.loss[0]
	else:
		raise Exception('unrecognized model type {0}'.format(opt.bert_type))
	
	m_hf.save_pretrained(opt.output)
	tokenizer.save_pretrained(opt.output)


if __name__ == '__main__':
	sys.exit(main(sys.argv[1:]))
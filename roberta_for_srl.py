import torch
from transformers import *
from transformers.modeling_roberta import RobertaPreTrainedModel
from crf_loss import *
from linear_classifier import *
from util import *


# RoBERTa for SRL, only meant for inference/demo
#	Training interface is not maintained here
class RobertaForSRL(RobertaPreTrainedModel):
	authorized_unexpected_keys = [r"pooler"]
	authorized_missing_keys = [r"position_ids"]

	def __init__(self, config, *model_args, **model_kwargs):
		super().__init__(config)

		# from json to dict lost track of data type, enforce types here
		config.label_map_inv = {int(k): v for k, v in config.label_map_inv.items()}

		# options can be overwritten by externally specified ones
		if 'overwrite_opt' in model_kwargs:
			for k, v in model_kwargs['overwrite_opt'].__dict__.items():
				setattr(config, k, v)
			for k, v in config.__dict__.items():
				setattr(model_kwargs['overwrite_opt'], k, v)

		self.num_labels = config.num_labels

		self.shared = model_kwargs['shared']
		self._loss_context = Holder()

		self.roberta = RobertaModel(config, add_pooling_layer=False)
		self.classifier = LinearClassifier(config, shared=self.shared)
		self.crf_loss = CRFLoss(opt=config, shared=self.shared)

		self.init_weights()

	# update the contextual info of current batch
	def update_context(self, orig_seq_l, sub2tok_idx, res_map=None):
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

	# shared: a namespace or a Holder instance that contains information for the current input batch
	#	such as, predicate labels, subtok to tok index mapping, etc
	def forward(self, input_ids):
		self.shared.batch_l = input_ids.shape[0]
		self.shared.seq_l = input_ids.shape[1]

		enc = self.roberta(input_ids)[0]

		log_pa, score, extra = self.classifier(enc)

		pred, _ = self.crf_loss.decode(log_pa, score)

		return pred
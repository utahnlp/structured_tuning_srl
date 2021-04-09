import torch
import transformers
from transformers import *
from loss.crf_loss import *
from modules.linear_classifier import *
from util.util import *
from packaging import version
if version.parse(transformers.__version__) < version.parse('4.0'):
	# for transformers 3+
	from transformers.modeling_roberta import RobertaPreTrainedModel
	from transformers.configuration_roberta import RobertaConfig
else:
	# for transformers 4+
	from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
	from transformers.models.roberta.configuration_roberta import RobertaConfig


# RoBERTa for SRL, only meant for inference/demo
#	Training interface is not maintained here
class RobertaForSRL(RobertaPreTrainedModel):
	authorized_unexpected_keys = [r"pooler"]
	authorized_missing_keys = [r"position_ids"]

	def __init__(self, config, *model_args, **model_kwargs):
		super().__init__(config)

		# the config and global opt should be handled better here, for now it's hacky
		# options can be overwritten by externally specified ones
		if 'global_opt' in model_kwargs:
			for k, v in model_kwargs['global_opt'].__dict__.items():
				setattr(config, k, v)
			for k, v in config.__dict__.items():
				setattr(model_kwargs['global_opt'], k, v)
		# explicitly trigger ad-hoc fix for options
		#	some options will have wrong data type after being loaded from config.json
		complete_opt(config)
		complete_opt(model_kwargs['global_opt'])

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

	# shared: a namespace or a Holder instance that contains information for the current input batch
	#	such as, predicate labels, subtok to tok index mapping, etc
	def forward(self, input_ids, v_label = None, v_l = None):
		self.shared.batch_l = input_ids.shape[0]
		self.shared.seq_l = input_ids.shape[1]

		enc = self.roberta(input_ids)[0]

		log_pa, score, extra = self.classifier(enc)

		pred, _ = self.crf_loss.decode(log_pa, score, v_label, v_l)

		return pred
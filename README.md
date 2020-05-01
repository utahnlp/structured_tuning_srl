<p align="center"><img width="80%" src="logo.png" /></p>

---

Implementation of our ACL 2020 paper: Structured Tuning for Semantic Role Labeling
```
@inproceedings{li2020structuredtuningsrl,
      author    = {Li, Tao and Jawale, Parth and Palmer, Martha and Srikumar, Vivek},
      title     = {Structured Tuning for Semantic Role Labeling},
      booktitle = {Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
      year      = {2020}
  }
```


## Prerequisites
```
perl	# for evaluation script
huggingface transformers
pip install sacremoses	# used by huggingface
nividia apex
and potentially more...
```

**CONLL 2005 and Preprocessing**
```
cd ./data/srl
wget http://www.lsi.upc.edu/~srlconll/conll05st-release.tar.gz
wget http://www.lsi.upc.edu/~srlconll/conll05st-tests.tar.gz
tar xf conll05st-release.tar.gz
tar xf conll05st-tests.tar.gz
# get perl dependency
wget https://www.cs.upc.edu/~srlconll/srlconll-1.1.tgz
tar xf srlconll-1.1.tgz
cd conll_extract/
./make_conll2005_data.sh ../data/treebank_3/

python3 preprocess.py --dir ./data/srl/ --batch_size 24 --bert_type roberta-base --train conll05.train.txt --val conll05.devel.txt --test1 conll05.test.wsj.txt --test2 conll05.test.brown.txt --tokenizer_output conll05 --output conll05
```

**CONLL 2012 and Preprocessing**
```
# generating from ontonotes 5.0 data
# 	get ontonotes 5.0 release of propbank
cd conll_extract/
./skeleton2conll.sh -D ../data/ontonotes-release-5.0/data/files/data/ ../data/srl/conll-formatted-ontonotes-5.0/
./make_conll2012_data.sh ../data/srl/conll-formatted-ontonotes-5.0/
# or get processed files from
cd ./data/
git clone https://github.com/yuchenlin/OntoNotes-5.0-NER-BIO.git
./make_conll2012_data.sh ../data/OntoNotes-5.0-NER-BIO/conll-formatted-ontonotes-5.0/

python3 preprocess.py --dir ./data/srl/ --batch_size 20 --bert_type roberta-base --max_seq_l 410 --max_num_v 45 --train conll2012.train.txt --val conll2012.devel.txt --test1 conll2012.test.txt --test2 "" --tokenizer_output conll2012 --output conll2012
```

**Propbank Frameset**

Extract and preprocess framesets for CONLL-2012 as an example:
```
python3 extract_frameset.py --dir ./data/propbank-frames/frames/ --output ./data/srl/frameset.txt
python3 preprocess_frameset.py --train conll2012.train.orig_tok_grouped.txt \
--val conll2012.val.orig_tok_grouped.txt --test1 conll2012.test1.orig_tok_grouped.txt \
--roleset_dict conll2012.roleset_id.dict --label_dict conll2012.label.dict --output conll2012
```


## Training on CONLL-05

```
mkdir models

GPUID=[GPUID]
DROP=0.5
BERT=base
HIDDEN=768
USE_GOLD=1
LR=0.00003
EPOCH=30
LOSS=crf
PERC=1
WARM=0.1
SEED=1
MODEL=./models/bert_${BERT}_${LOSS}_lr${LR//.}_drop${DROP//.}_gold${USE_GOLD}_epoch${EPOCH}_seed${SEED}_perc${PERC//.}
python3 -u train.py --gpuid $GPUID --bert_gpuid $GPUID --dir ./data/srl/ --train_data conll05.train.hdf5 --val_data conll05.val.hdf5 \
	--train_res conll05.train.orig_tok_grouped.txt,conll05.train.frame.hdf5,conll05.frame_pool.hdf5 \
	--val_res conll05.val.orig_tok_grouped.txt,conll05.val.frame.hdf5,conll05.frame_pool.hdf5 \
	 --label_dict conll05.label.dict --num_frame 39 \
	--loss $LOSS --optim adamw_fp16 --epochs $EPOCH --warmup_perc $WARM --learning_rate $LR --dropout $DROP --compact_mode whole_word \
	--bert_type roberta-${BERT} --bert_size $HIDDEN --hidden_size $HIDDEN --use_gold_predicate $USE_GOLD \
	--percent $PERC --val_percent 1 --seed $SEED --conll_output $MODEL --save_file $MODEL | tee ${MODEL}.txt
done
```

**2nd round of finetuning**

```

GPUID=[GPUID]
DROP=0.5
BERT=base
HIDDEN=768
USE_GOLD=1
LR=0.00001
EPOCH=5
LOSS=crf,unique_role,frame_role,overlap_role
SEED=1
PERC=1
WARM=0.1
LAMBD=1,1,0.5,0.1
LOAD=./models/bert_${BERT}_crf_lr000003_drop05_gold${USE_GOLD}_epoch30_seed${SEED}_perc${PERC//.}
MODEL=./models/bert2_${BERT}_${LOSS//,}_lambd${LAMBD//.}_lr${LR//.}_drop${DROP//.}_gold${USE_GOLD}_epoch${EPOCH}_seed${SEED}_perc${PERC//.}
python3 -u train.py --gpuid $GPUID --bert_gpuid $GPUID --dir ./data/srl/ --train_data conll05.train.hdf5 --val_data conll05.val.hdf5 \
	--train_res conll05.train.orig_tok_grouped.txt,conll05.train.frame.hdf5,conll05.frame_pool.hdf5 \
	--val_res conll05.val.orig_tok_grouped.txt,conll05.val.frame.hdf5,conll05.frame_pool.hdf5 \
	--label_dict conll05.label.dict --num_frame 39 \
	--optim adamw_fp16 --epochs $EPOCH --warmup_perc $WARM --learning_rate $LR --dropout $DROP --compact_mode whole_word \
	--bert_type roberta-${BERT} --bert_size $HIDDEN --hidden_size $HIDDEN --use_gold_predicate $USE_GOLD \
	--loss $LOSS --lambd $LAMBD \
	--load $LOAD \
	--percent $PERC --val_percent 1 --seed $SEED --conll_output ${MODEL} --save_file $MODEL | tee ${MODEL}.txt
```


**Evaluation**
```

GPUID=[GPUID]
DROP=0.5
BERT=base
HIDDEN=768
USE_GOLD=1
LR=0.00001
EPOCH=5
LOSS=crf,unique_role,frame_role,overlap_role
LAMBD=1,1,0.5,0.1
SEED=1
PERC=1
TEST=test1
MODEL=./models/bert2_${BERT}_${LOSS//,}_lambd${LAMBD//.}_lr${LR//.}_drop${DROP//.}_gold${USE_GOLD}_epoch${EPOCH}_seed${SEED}_perc${PERC//.}
python3 -u eval.py --gpuid $GPUID --bert_gpuid $GPUID --dir ./data/srl/ --data conll05.${TEST}.hdf5 \
	--res conll05.${TEST}.orig_tok_grouped.txt,conll05.${TEST}.frame.hdf5,conll05.frame_pool.hdf5 \
	--label_dict conll05.label.dict --num_frame 39 \
	--dropout 0 --compact_mode whole_word \
	--bert_type roberta-${BERT} --bert_size $HIDDEN --hidden_size $HIDDEN --use_gold_predicate $USE_GOLD \
	--loss $LOSS --lambd $LAMBD \
	--conll_output ${MODEL} --load_file ${MODEL} | tee ${MODEL}.testlog.txt

perl srl-eval.pl ${MODEL}.gold.txt ${MODEL}.pred.txt
```
where ``TEST=test1`` is for WSJ set. Set ``TEST=test2`` to evaluate on Brown set.

## Training on CONLL-2012

```

GPUID=[GPUID]
DROP=0.5
BERT=base
HIDDEN=768
USE_GOLD=1
LR=0.00003
EPOCH=30
LOSS=crf
PERC=1
VAL_PERC=1
WARM=0.1
SEED=1
MODEL=./models/bert2012_${BERT}_${LOSS//,}_lr${LR//.}_drop${DROP//.}_gold${USE_GOLD}_epoch${EPOCH}_seed${SEED}_perc${PERC//.}
python3 -u train.py --gpuid $GPUID --bert_gpuid $GPUID --dir ./data/srl/ --train_data conll2012.train.hdf5 --val_data conll2012.val.hdf5 \
	--train_res conll2012.train.orig_tok_grouped.txt,conll2012.train.frame.hdf5,conll2012.frame_pool.hdf5 \
	--val_res conll2012.val.orig_tok_grouped.txt,conll2012.val.frame.hdf5,conll2012.frame_pool.hdf5 \
	--label_dict conll2012.label.dict \
	--loss $LOSS --optim adamw_fp16 --epochs $EPOCH --warmup_perc $WARM --learning_rate $LR --dropout $DROP --compact_mode whole_word \
	--bert_type roberta-${BERT} --bert_size $HIDDEN --hidden_size $HIDDEN --use_gold_predicate $USE_GOLD \
	--num_label 129 --percent $PERC --val_percent $VAL_PERC \
	--seed $SEED --conll_output $MODEL --save_file $MODEL | tee ${MODEL}.txt
```

**2nd round of finetuning**

```

GPUID=[GPUID]
DROP=0.5
BERT=base
HIDDEN=768
USE_GOLD=1
LR=0.00001
EPOCH=5
PERC=1
VAL_PERC=1
WARM=0.1
LOSS=crf,unique_role,frame_role,overlap_role
LAMBD=1,1,1,0.1
SEED=1
LOAD=./models/bert2012_${BERT}_crf_lr000003_drop${DROP//.}_gold${USE_GOLD}_epoch30_seed${SEED}_perc${PERC//.}
MODEL=./models/bert2_2012_${BERT}_${LOSS//,}_lambd${LAMBD//.}_lr${LR//.}_drop${DROP//.}_gold${USE_GOLD}_epoch${EPOCH}_seed${SEED}_perc${PERC//.}
python3 -u train.py --gpuid $GPUID --bert_gpuid $GPUID --dir ./data/srl/ --train_data conll2012.train.hdf5 --val_data conll2012.val.hdf5 \
	--train_res conll2012.train.orig_tok_grouped.txt,conll2012.train.frame.hdf5,conll2012.frame_pool.hdf5 \
	--val_res conll2012.val.orig_tok_grouped.txt,conll2012.val.frame.hdf5,conll2012.frame_pool.hdf5 \
	--label_dict conll2012.label.dict \
	--optim adamw_fp16 --epochs $EPOCH --warmup_perc $WARM --learning_rate $LR --dropout $DROP --compact_mode whole_word \
	--bert_type roberta-${BERT} --bert_size $HIDDEN --hidden_size $HIDDEN --use_gold_predicate $USE_GOLD \
	--num_label 129 --percent $PERC --val_percent $VAL_PERC --loss $LOSS --lambd $LAMBD \
	--load $LOAD \
	--seed $SEED --conll_output ${MODEL} --save_file $MODEL | tee ${MODEL}.txt
```

**Evaluation**

```

GPUID=0
DROP=0.5
BERT=base
HIDDEN=768
USE_GOLD=1
LR=0.00001
EPOCH=5
SEED=1
PERC=1
LOSS=crf,unique_role,frame_role,overlap_role
LAMBD=1,1,1,0.1
TEST=test1
MODEL=./models/bert2_2012_${BERT}_${LOSS//,}_lambd${LAMBD//.}_lr${LR//.}_drop${DROP//.}_gold${USE_GOLD}_epoch${EPOCH}_seed${SEED}_perc${PERC//.}
python3 -u eval.py --gpuid $GPUID --bert_gpuid $GPUID --dir ./data/srl/ --data conll2012.${TEST}.hdf5 \
--res conll2012.${TEST}.orig_tok_grouped.txt,conll2012.${TEST}.frame.hdf5,conll2012.frame_pool.hdf5 \
--label_dict conll2012.label.dict \
--dropout 0 --compact_mode whole_word \
--bert_type roberta-base --bert_size $HIDDEN --hidden_size $HIDDEN --use_gold_predicate $USE_GOLD \
--num_label 129 --loss $LOSS  --lambd $LAMBD \
--conll_output ${MODEL} --load_file ${MODEL} | tee ${MODEL}.testlog.txt

perl srl-eval.pl ${MODEL}.gold.txt ${MODEL}.pred.txt
```


## To-dos
- [ ] Sanity check

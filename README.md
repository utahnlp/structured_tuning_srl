<p align="center"><img width="80%" src="logo.png" /></p>

---

Implementation of our ACL 2020 paper: [Structured Tuning for Semantic Role Labeling](https://arxiv.org/abs/2005.00496)
```
@inproceedings{li2020structuredtuningsrl,
      author    = {Li, Tao and Jawale, Parth Anand and Palmer, Martha and Srikumar, Vivek},
      title     = {Structured Tuning for Semantic Role Labeling},
      booktitle = {Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics},
      year      = {2020}
  }
```
---

In addition to dependencies in ``requirements.txt``, please install perl for evaluation and [Nvidia-apex](https://github.com/NVIDIA/apex) for GPU speedup.

The flow of this readme is:
* [Preprocessing](#preprocessing): dataset preprocessing
* [CoNLL-05](#conll05): Training and evlauation for CoNLL-05 dataset
* [CoNLL-2012](#conll2012): Training and evlauation for CoNLL-2012 dataset
* [Demo](#demo): A very easy-to-use **demo** that fetches trained model(s) to interactively predict on user inputs
---

<a name="preprocessing"></a>
# Preprocessing

**Extracting Propbank Frameset**

First make sure propbank frames are downloaded and extracted to ``./data/propbank-frames/frames/``.
Then extract framesets:
```
python3 -u -m preprocess.extract_frameset --dir ./data/propbank-frames/frames/ --output ./data/srl/frameset.txt
```

**Preprocessing CoNLL 2005**
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

python3 -u -m preprocess.preprocess --dir ./data/srl/ \
	--batch_size 24 --bert_type roberta-base \
	--train conll05.train.txt --val conll05.devel.txt --test1 conll05.test.wsj.txt --test2 conll05.test.brown.txt \
	--tokenizer_output conll05 --output conll05
```

**Preprocessing CoNLL 2012**
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

python3 -u -m preprocess.preprocess --dir ./data/srl/ \
	--batch_size 20 --bert_type roberta-base --max_seq_l 410 --max_num_v 45 \
	--train conll2012.train.txt --val conll2012.devel.txt --test1 conll2012.test.txt --test2 "" \
	--tokenizer_output conll2012 --output conll2012
```

**Preprocessing Frameset for CoNLL-05 and CoNLL-2012**

To preprocess framesets for CoNLL-05:
```
python3 -u -m preprocess.preprocess_frameset --roleset_dict conll05.roleset_id.dict --label_dict conll05.label.dict \
	--train conll05.train.orig_tok_grouped.txt --val conll05.val.orig_tok_grouped.txt \
	--test1 conll05.test1.orig_tok_grouped.txt --test2 conll05.test2.orig_tok_grouped.txt \
	--output conll05
```

To preprocess framesets for CoNLL-2012:
```
python3 -u -m preprocess.preprocess_frameset --train conll2012.train.orig_tok_grouped.txt \
	--val conll2012.val.orig_tok_grouped.txt --test1 conll2012.test1.orig_tok_grouped.txt \
	--roleset_dict conll2012.roleset_id.dict --label_dict conll2012.label.dict --output conll2012
```

<a name="conll05"></a>
# Training and Evaluation on CoNLL-05

```
mkdir models

GPUID=[GPUID]
DROP=0.5
LR=0.00003
EPOCH=30
LOSS=crf
PERC=1
SEED=1
MODEL=./models/roberta_base_${LOSS}_lr${LR//.}_drop${DROP//.}_gold1_epoch${EPOCH}_seed${SEED}_perc${PERC//.}
python3 -u train.py --gpuid $GPUID --dir ./data/srl/ --train_data conll05.train.hdf5 --val_data conll05.val.hdf5 \
	--train_res conll05.train.orig_tok_grouped.txt,conll05.train.frame.hdf5,conll05.frame_pool.hdf5 \
	--val_res conll05.val.orig_tok_grouped.txt,conll05.val.frame.hdf5,conll05.frame_pool.hdf5 \
	--label_dict conll05.label.dict \
	--bert_type roberta-base --loss $LOSS --epochs $EPOCH --learning_rate $LR --dropout $DROP  \
	--percent $PERC --seed $SEED \
	--conll_output $MODEL --save_file $MODEL | tee ${MODEL}.txt
```
where ``[GPUID]`` is the GPU device index.

**2nd round of finetuning**

```
GPUID=[GPUID]
DROP=0.5
LR=0.00001
EPOCH=5
LOSS=crf,unique_role,frame_role,overlap_role
SEED=1
PERC=1
LAMBD=1,1,0.5,0.1
LOAD=./models/roberta_base_crf_lr000003_drop05_gold1_epoch30_seed${SEED}_perc${PERC//.}
MODEL=./models/roberta2_base_${LOSS//,}_lambd${LAMBD//.}_lr${LR//.}_drop${DROP//.}_gold1_epoch${EPOCH}_seed${SEED}_perc${PERC//.}
python3 -u train.py --gpuid $GPUID --dir ./data/srl/ --train_data conll05.train.hdf5 --val_data conll05.val.hdf5 \
	--train_res conll05.train.orig_tok_grouped.txt,conll05.train.frame.hdf5,conll05.frame_pool.hdf5 \
	--val_res conll05.val.orig_tok_grouped.txt,conll05.val.frame.hdf5,conll05.frame_pool.hdf5 \
	--label_dict conll05.label.dict \
	--bert_type roberta-base --loss $LOSS --epochs $EPOCH --learning_rate $LR --dropout $DROP --lambd $LAMBD \
	--percent $PERC --seed $SEED \
	--load $LOAD --conll_output ${MODEL} --save_file $MODEL | tee ${MODEL}.txt
```


**Evaluation**
```
GPUID=[GPUID]
DROP=0.5
LR=0.00001
EPOCH=5
LOSS=crf,unique_role,frame_role,overlap_role
LAMBD=1,1,0.5,0.1
SEED=1
PERC=1
TEST=test1
MODEL=./models/roberta2_base_${LOSS//,}_lambd${LAMBD//.}_lr${LR//.}_drop${DROP//.}_gold1_epoch${EPOCH}_seed${SEED}_perc${PERC//.}
python3 -u eval.py --gpuid $GPUID --dir ./data/srl/ --data conll05.${TEST}.hdf5 \
	--res conll05.${TEST}.orig_tok_grouped.txt,conll05.${TEST}.frame.hdf5,conll05.frame_pool.hdf5 \
	--label_dict conll05.label.dict \
	--bert_type roberta-base --loss $LOSS --lambd $LAMBD \
	--load_file ${MODEL} --conll_output ${MODEL} | tee ${MODEL}.testlog.txt

perl srl-eval.pl ${MODEL}.gold.txt ${MODEL}.pred.txt
```
where ``TEST=test1`` is for WSJ set. Set ``TEST=test2`` to evaluate on Brown set.


<a name="conll2012"></a>
# Training and Evlauation on CoNLL-2012

```
GPUID=[GPUID]
DROP=0.5
USE_GOLD=1
LR=0.00003
EPOCH=30
LOSS=crf
PERC=1
SEED=1
MODEL=./models/roberta_base_2012_${LOSS//,}_lr${LR//.}_drop${DROP//.}_gold1_epoch${EPOCH}_seed${SEED}_perc${PERC//.}
python3 -u train.py --gpuid $GPUID --dir ./data/srl/ --train_data conll2012.train.hdf5 --val_data conll2012.val.hdf5 \
	--train_res conll2012.train.orig_tok_grouped.txt,conll2012.train.frame.hdf5,conll2012.frame_pool.hdf5 \
	--val_res conll2012.val.orig_tok_grouped.txt,conll2012.val.frame.hdf5,conll2012.frame_pool.hdf5 \
	--label_dict conll2012.label.dict \
	--bert_type roberta-base --loss $LOSS  --epochs $EPOCH --learning_rate $LR --dropout $DROP \
	--percent $PERC --seed $SEED \
	--conll_output $MODEL --save_file $MODEL | tee ${MODEL}.txt
```

**2nd round of finetuning**

```
GPUID=[GPUID]
DROP=0.5
LR=0.00001
EPOCH=5
PERC=1
LOSS=crf,unique_role,frame_role,overlap_role
LAMBD=1,1,1,0.1
SEED=1
LOAD=./models/roberta_base_2012_crf_lr000003_drop${DROP//.}_gold1_epoch30_seed${SEED}_perc${PERC//.}
MODEL=./models/roberta2_base_2012_${LOSS//,}_lambd${LAMBD//.}_lr${LR//.}_drop${DROP//.}_gold1_epoch${EPOCH}_seed${SEED}_perc${PERC//.}
python3 -u train.py --gpuid $GPUID --dir ./data/srl/ --train_data conll2012.train.hdf5 --val_data conll2012.val.hdf5 \
	--train_res conll2012.train.orig_tok_grouped.txt,conll2012.train.frame.hdf5,conll2012.frame_pool.hdf5 \
	--val_res conll2012.val.orig_tok_grouped.txt,conll2012.val.frame.hdf5,conll2012.frame_pool.hdf5 \
	--label_dict conll2012.label.dict \
	--bert_type roberta-base --loss $LOSS --epochs $EPOCH --learning_rate $LR --dropout $DROP --lambd $LAMBD \
	--percent $PERC --seed $SEED \
	--load $LOAD --conll_output ${MODEL} --save_file $MODEL | tee ${MODEL}.txt
```

**Evaluation**

```
GPUID=0
DROP=0.5
LR=0.00001
EPOCH=5
SEED=1
PERC=1
LOSS=crf,unique_role,frame_role,overlap_role
LAMBD=1,1,1,0.1
TEST=test1
MODEL=./models/roberta2_base_2012_${LOSS//,}_lambd${LAMBD//.}_lr${LR//.}_drop${DROP//.}_gold1_epoch${EPOCH}_seed${SEED}_perc${PERC//.}
python3 -u eval.py --gpuid $GPUID --dir ./data/srl/ --data conll2012.${TEST}.hdf5 \
	--res conll2012.${TEST}.orig_tok_grouped.txt,conll2012.${TEST}.frame.hdf5,conll2012.frame_pool.hdf5 \
	--label_dict conll2012.label.dict \
	--bert_type roberta-base --loss $LOSS --lambd $LAMBD \
	--load_file ${MODEL} --conll_output ${MODEL} | tee ${MODEL}.testlog.txt

perl srl-eval.pl ${MODEL}.gold.txt ${MODEL}.pred.txt
```

<a name="demo"></a>
# Demo
You can use a trained model to do inference interactively:
```
python3 -u -m hf.demo --load_file tli8hf/robertabase-crf-conll2012 --gpuid [GPUID]
```
where ``[GPUID]`` is the GPU device index. Set it to ``-1`` to run on CPU.

The demo will automatically download a RoBERTa+CRF model pre-trained on the CoNLL2012 data, and use it for interactive prediction.
Available models are:

| Model    |  --load_file  |  CoNLL-2012 test F1  |
|----------|:-------------|:------|
| RoBERTa+CRF |  tli8hf/robertabase-crf-conll2012 | 85.9* |
| RoBERTa+U,F,O |  tli8hf/robertabase-structured-tuning-srl-conll2012   | 86.6 |

F1 scores with ``*``: trained and evaluated without gold predicate (i.e. ``--use_gold_predicate 0``).

# Acknowledgements
- [x] Sanity check (Thanks to Ghazaleh Kazeminejad for helping me with sanity check)

# TODO
- [x] Upload more models to HuggingFace hub
- [ ] extend demo interface to accept predicate
- [x] Make a separate predicate classifier

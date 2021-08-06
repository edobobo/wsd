# WSD
Repo for transformers-wsd and eventually ESC

## Supported Frameworks

### Raganato

**Example**: train *simple_transformer* on *SemCor* with early stopping on *SemEval-2007* F1 score:

```bash
PYTHONPATH=$(pwd) python src/scripts/model/train.py \
  train=simple_transformer \
  train.model_name=simple-semcor \
  data=simple_transformer \
  data.train_dataset.disambiguation_corpus.raganato_path=data/WSD_Evaluation_Framework/Training_Corpora/SemCor/semcor \
  data.validation_dataset.disambiguation_corpus.raganato_path=data/WSD_Evaluation_Framework/Evaluation_Datasets/semeval2007/semeval2007 \
  model=simple_transformer \
  model.transformer_model=bert-base-cased \
  tokenizer=simple_transformer \
  callbacks=simple_transformer
```

**Example**: evaluate the trained model on the *ALL* dataset:

```bash
PYTHONPATH=$(pwd) python src/scripts/eval/raganato_evaluate.py \
  --config-name simple_transformer_raganato \
  raganato_path=data/WSD_Evaluation_Framework/Evaluation_Datasets/ALL/ALL \
  module_ckpt=<model folder>/checkpoints/best.ckpt \
  predictor.sense_vocabulary_path=<model folder>/sense_vocabulary.txt
```

### XL-WSD

```bash
XL_WSD_LANGUAGE=it PYTHONPATH=$(pwd) python src/scripts/model/train.py \
  train=simple_transformer \
  train.model_name=xlwsd-it-mbert-fb \
  data=simple_transformer_xlwsd \
  data.train_dataset.disambiguation_corpus.raganato_path=data/xl-wsd/training_datasets/semcor_it/semcor_it \
  model=simple_transformer \
  model.transformer_model=bert-base-multilingual-cased \
  model.text_encoder.fine_tune=False \
  tokenizer=simple_transformer \
  callbacks=simple_transformer_xlwsd
```

```bash
XL_WSD_LANGUAGE=it PYTHONPATH=$(pwd) python src/scripts/eval/xlwsd_evaluate.py \
  --config-name simple_transformer_xlwsd \
  module_ckpt=<model folder>/checkpoints/best.ckpt \
  predictor.sense_vocabulary_path=<model folder>/sense_vocabulary.txt
```
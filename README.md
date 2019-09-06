# mimic-phenotype-classification

This repository contains the code for the downstream classification task. Follow the steps below:


1. Install the environment
```
conda env create -f environment.yml
```
2. Run the files sequentially. This assumes you have already run the text generation models and generated synthetic data
3. You may find it useful to run the training script as follows:
```
nohup python 02_train.py > train.out &
```
4. You must install BioBERT separately as instructed below

### BioBERT

In order to be able to use BioBERT, follow these instructions:

1. Download the BioBERT v1.1 (+ PubMed 1M) model (or any other model) from the [BioBERT repo](https://github.com/naver/biobert-pretrained)
2. Extract the downloaded file, e.g. with `tar -xvzf biobert_v1.1_pubmed.tar.gz`
3. Convert the BioBERT model TensorFlow checkpoint to a PyTorch and PyTorch-Transformers compatible one: `pytorch_transformers bert biobert_v1.1_pubmed/model.ckpt-1000000 biobert_v1.1_pubmed/bert_config.json biobert_v1.1_pubmed/pytorch_model.bin`
4. Rename config `mv biobert_v1.1_pubmed/bert_config.json biobert_v1.1_pubmed/config.json`
5. Rename folder `mv biobert_v1.1_pubmed/ biobert/`

If everything has worked, this python code snippet shouldn't give any errors:

```python
from pytorch_transformers import BertModel
model = BertModel.from_pretrained('biobert')
```

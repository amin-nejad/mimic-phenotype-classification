# mimic-phenotype-classification

Working repository for Master's project. Downstream task for artificial data created in [mimic-text-generation](https://github.com/amin-nejad/mimic-text-generation)

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

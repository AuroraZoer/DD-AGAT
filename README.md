# DD-AGAT

This is the implementation of `Integrating Dependency Type and Directionality into Adapted Graph Attention Networks to Enhance Relation Extraction` at ICDAR 2024.

You can e-mail **Yiran Zhao** at **ZhaoYiran@emails.bjut.edu.cn**, if you have any questions.

## Citation

If you want to use our codes and datasets in your research, please cite:

```
```

## Requirements

Our code works with the following environment.

- `python>=3.7`

- `pytorch>=1.3`

## Dataset and Data Processing

To obtain the data, you can go to [`data`](https://github.com/cuhksz-nlp/RE-AGCN/blob/main/data) directory for details.

## Downloading BERT

In our paper, we use BERT ([paper](https://www.aclweb.org/anthology/N19-1423/)) as the encoder.

For BERT, please download pre-trained BERT-Base English from [Google](https://github.com/google-research/bert) or from [HuggingFace](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz). If you download it from Google, you need to convert the model from TensorFlow version to PyTorch version.

if you choose to download the model directly from HuggingFace, click the link below:
- [bert-base-uncased](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz)

- [bert-large-uncased](https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased.tar.gz)

- [bert-base-cased](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz)

- [bert-large-cased](https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased.tar.gz)

- [bert-base-multilingual-uncased](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased.tar.gz)

- [bert-base-multilingual-cased](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz)

- [bert-base-chinese](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz)


## Training and Testing

You can find the command lines to train and test models in `run_train.sh` and `run_test.sh`, respectively.

Here are some important parameters:

- `--do_train`: train the model.
- `--do_eval`: test the model.

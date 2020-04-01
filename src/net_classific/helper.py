# -*- coding: utf-8 -*-#

# -------------------------------------------------------------------------------
# Name:         helper
# Description:  
# Author:       zjjhit
# Date:         2020/3/31
# -------------------------------------------------------------------------------

import torch

BATCH_SIZE = 16
NGRAMS = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_batch(batch):
    label = torch.tensor([entry[0] for entry in batch])
    text = [entry[1] for entry in batch]
    offsets = [0] + [len(entry) for entry in text]
    # torch.Tensor.cumsum returns the cumulative sum
    # of elements in the dimension dim.
    # torch.Tensor([1.0, 2.0, 3.0]).cumsum(dim=0)

    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text = torch.cat(text)
    return text, offsets, label


import logging
from torchtext.datasets.text_classification import build_vocab_from_iterator
from torchtext.datasets.text_classification import _csv_iterator
from torchtext.datasets.text_classification import extract_archive
from torchtext.datasets.text_classification import _create_data_from_iterator
from torchtext.datasets.text_classification import TextClassificationDataset


def make_data(path_root='../data/ag_news_csv.tgz', ngrams=2, vocab=None, include_unk=False):
    extracted_files = extract_archive(path_root)

    for fname in extracted_files:
        if fname.endswith('train.csv'):
            train_csv_path = fname
        if fname.endswith('test.csv'):
            test_csv_path = fname

    vocab = build_vocab_from_iterator(_csv_iterator(train_csv_path, ngrams))

    logging.info('Vocab has {} entries'.format(len(vocab)))
    logging.info('Creating training data')

    train_data, train_labels = _create_data_from_iterator(
        vocab, _csv_iterator(train_csv_path, ngrams, yield_cls=True), include_unk)
    logging.info('Creating testing data')
    test_data, test_labels = _create_data_from_iterator(
        vocab, _csv_iterator(test_csv_path, ngrams, yield_cls=True), include_unk)
    if len(train_labels ^ test_labels) > 0:
        raise ValueError("Training and test labels don't match")
    return (TextClassificationDataset(vocab, train_data, train_labels),
            TextClassificationDataset(vocab, test_data, test_labels))




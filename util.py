import os
import sys
import json
import torch
import pyhocon
import coref_ops
import dataloader
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer
from torch.utils.data import DataLoader

def flatten(l):
  return [item for sublist in l for item in sublist]

def truncated_normal_(tensor, mean=0, std=1):
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor

def get_top_span_indices(candidate_mention_scores, candidate_starts, candidate_ends, num_words, k):
    tf_candidate_mention_scores = tf.convert_to_tensor(candidate_mention_scores.clone().detach().numpy())
    tf_candidate_starts = tf.convert_to_tensor(candidate_starts.clone().detach().numpy(), dtype='int32')
    tf_candidate_ends = tf.convert_to_tensor(candidate_ends.clone().detach().numpy(), dtype='int32')
    return coref_ops.extract_spans(tf.expand_dims(tf_candidate_mention_scores, 0), 
  								 tf.expand_dims(tf_candidate_starts, 0),
  								 tf.expand_dims(tf_candidate_ends, 0),
  								 tf.expand_dims(k, 0),
  								 num_words,
  								 True) # [1, k]

def initialize_from_env(eval_test=False):

    name = sys.argv[1]
    print("Running experiment: {}".format(name))

    if eval_test:
      config = pyhocon.ConfigFactory.parse_file("test.experiments.conf")[name]
    else:
      config = pyhocon.ConfigFactory.parse_file("experiments.conf")[name]
    config["log_dir"] = mkdirs(os.path.join(config["log_root"], name))

    print(pyhocon.HOCONConverter.convert(config, "hocon"))
    return config

def mkdirs(path):
    try:
      os.makedirs(path)
    except OSError as exception:
      if exception.errno != errno.EEXIST:
        raise
    return path

# convert into torch tensor
def collate_fn(example):
    example = example[0]
    return {"doc_key": example[0],
            "input_ids": torch.tensor(example[1]).long(),
            "input_mask": torch.tensor(example[2]).long(),
            "clusters": example[3],
            "text_len": example[4],
            "speaker_ids": torch.tensor(example[5]).long(),
            "genre": example[6],
            "gold_starts": torch.tensor(example[7]).long(),
            "gold_ends": torch.tensor(example[8]).long(),
            "cluster_ids": torch.tensor(example[9], dtype=torch.int32),
            "sentence_map": example[10]
            }

def get_train_dataloader(config):
    examples = []
    with open(config["train_path"],"r") as file:
      for line in file.readlines():
        examples.append(json.loads(line))

    tokenizer = BertTokenizer.from_pretrained(config["model_name"], do_lower_case=True)
    dataset = dataloader.MyDataset(examples, config["max_segment_len"], config["max_training_sentences"], config["genres"], tokenizer, True)
    train_dataloader = DataLoader(dataset, batch_size=1, drop_last=False, shuffle=False, collate_fn=collate_fn)
    return train_dataloader
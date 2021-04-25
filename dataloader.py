import sys
import util
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader


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

class MyDataset(Dataset):

    def __init__(self, data, max_segment_length, max_training_segments, genre, tokenizer, is_training):
        self.examples = data
        self.max_segment_length = max_segment_length
        self.max_training_segments = max_training_segments
        self.genres = { g:i for i,g in enumerate(genre) }
        self.is_training = is_training
        self.tokenizer = tokenizer
        self.subtoken_maps = {}

    def __len__(self):
        return len(self.examples)
    
    def get_speaker_dict(self, speakers):
        speaker_dict = {'UNK': 0, '[SPL]': 1}
        for s in speakers:
          if s not in speaker_dict and len(speaker_dict) < 20:
            speaker_dict[s] = len(speaker_dict)
        return speaker_dict

    def tensorize_mentions(self, mentions):
        if len(mentions) > 0:
          starts, ends = zip(*mentions)
        else:
          starts, ends = [], []
        return np.array(starts), np.array(ends)
    
    def truncate_example(self, doc_key, input_ids, input_mask, clusters, text_len, speaker_ids, genre, gold_starts, gold_ends, cluster_ids, sentence_map, sentence_offset=None):
        max_training_sentences = 8
        num_sentences = input_ids.shape[0]
        assert num_sentences > max_training_sentences

        sentence_offset = random.randint(0, num_sentences - max_training_sentences) if sentence_offset is None else sentence_offset
        word_offset = text_len[:sentence_offset].sum()
        num_words = text_len[sentence_offset:sentence_offset + max_training_sentences].sum()
        input_ids = input_ids[sentence_offset:sentence_offset + max_training_sentences, :]
        input_mask = input_mask[sentence_offset:sentence_offset + max_training_sentences, :]
        speaker_ids = speaker_ids[sentence_offset:sentence_offset + max_training_sentences, :]
        text_len = text_len[sentence_offset:sentence_offset + max_training_sentences]

        sentence_map = sentence_map[word_offset: word_offset + num_words]
        gold_spans = np.logical_and(gold_ends >= word_offset, gold_starts < word_offset + num_words)
        gold_starts = gold_starts[gold_spans] - word_offset
        gold_ends = gold_ends[gold_spans] - word_offset
        cluster_ids = cluster_ids[gold_spans]

        return doc_key, input_ids, input_mask, clusters, text_len, speaker_ids, genre,  gold_starts, gold_ends, cluster_ids, sentence_map


    def __getitem__(self, idx):
        example = self.examples[idx]
        # transform the clusters and mentions
        clusters = example['clusters']
        gold_mentions = sorted(tuple(m) for m in util.flatten(clusters))
        gold_mention_map = {m:i for i,m in enumerate(gold_mentions)}
        cluster_ids = np.zeros(len(gold_mentions))
        for cluster_id, cluster in enumerate(clusters):
          for mention in cluster:
            cluster_ids[gold_mention_map[tuple(mention)]] = cluster_id + 1
        
        # transform input_ids, mask, speaker
        sentences = example["sentences"]
        num_words = sum(len(s) for s in sentences)
        speakers = example["speakers"]
        speaker_dict = self.get_speaker_dict(util.flatten(speakers))
        sentence_map = example['sentence_map']
        text_len = np.array([len(s) for s in sentences])
        input_ids, input_mask, speaker_ids = [], [], []
        for i, (sentence, speaker) in enumerate(zip(sentences, speakers)):
          sent_input_ids = self.tokenizer.convert_tokens_to_ids(sentence)
          sent_input_mask = [1] * len(sent_input_ids)
          sent_speaker_ids = [speaker_dict.get(s, 3) for s in speaker]
          if len(sent_input_ids) < self.max_segment_length:
              padding = [0] * (self.max_segment_length - text_len[i])
              sent_input_ids = sent_input_ids + padding
              sent_input_mask = sent_input_mask + padding
              sent_speaker_ids = sent_speaker_ids + padding
          input_ids.append(sent_input_ids)
          speaker_ids.append(sent_speaker_ids)
          input_mask.append(sent_input_mask)
        input_ids = np.array(input_ids)
        input_mask = np.array(input_mask)
        speaker_ids = np.array(speaker_ids)
        assert num_words == np.sum(input_mask)

        #get genre idx and position
        doc_key = example["doc_key"]
        self.subtoken_maps[doc_key] = example.get("subtoken_map", None)
        genre = self.genres.get(doc_key[:2], 0)
        gold_starts, gold_ends = self.tensorize_mentions(gold_mentions)
        example_tensors = (doc_key, input_ids, input_mask, clusters, text_len, speaker_ids, genre,
                           gold_starts, gold_ends, cluster_ids, sentence_map)
        
        # randomly select max_training_segments and corresponding clusters, mentions
        if self.is_training and len(sentences) > self.max_training_segments:
             return self.truncate_example(*example_tensors)
        else:
             return example_tensors
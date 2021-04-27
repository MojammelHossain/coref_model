import math
import util
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from transformers import BertModel


class BertForCoref(nn.Module):
  def __init__(self, config):
    super(BertForCoref, self).__init__()
    self.config = config
    self.bert = BertModel.from_pretrained(config['model_name'])
    self.mention_word_attn = nn.Linear(768, 1, bias=True)
    self.in_features = 1536
    self.coref_layer_in = 0
    if config['model_heads']:
      self.in_features = 2304
    if config['use_features']:
      self.span_width_embedding = nn.Parameter(util.truncated_normal_(torch.zeros([config['max_span_width'], config['feature_size']]), std=0.02), requires_grad=True)
      self.coref_layer_antecedent_distance_embedding = nn.Parameter(util.truncated_normal_(torch.zeros([10, config['feature_size']]), std=0.02), requires_grad=True)
      self.in_features += config['feature_size']
      self.coref_layer_in += config['feature_size']
    self.mention_span_scores = self.get_layer(self.in_features, config['ffnn_depth'], config['ffnn_size'], 1)
    
    if config['use_prior']:
      self.span_width_prior_embedding = nn.Parameter(util.truncated_normal_(torch.zeros([config['max_span_width'], config['feature_size']]), std=0.02), requires_grad=True)
      self.mention_width_scores = self.get_layer(config['feature_size'], config['ffnn_depth'], config['ffnn_size'], 1)
      self.antecedent_distance_embedding = nn.Parameter(util.truncated_normal_(torch.zeros([10, config['feature_size']]), std=0.02), requires_grad=True)
      self.distance_scores_linear = nn.Linear(config['feature_size'], 1, bias=True)
    
    self.genre_embedding = nn.Parameter(util.truncated_normal_(torch.zeros([len(config['genres']), config['feature_size']]), std=0.02), requires_grad=True)
    self.src_projection = nn.Linear(self.in_features, self.in_features, bias=True)
    if config['use_metadata']:
      self.coref_layer_speaker_embedding = nn.Parameter(util.truncated_normal_(torch.zeros([2, config['feature_size']]),std=0.02), requires_grad=True)
      self.coref_layer_in += (config['feature_size']*2)
    if config['use_segment_distance']:
      self.coref_layer_in += config['feature_size']
      self.coref_layer_segment_distance_embedding = nn.Parameter(util.truncated_normal_(torch.zeros([self.config['max_training_sentences'], self.config['feature_size']]), std=0.02), requires_grad=True)
    self.coref_layer_slow_antecedent_scores = self.get_layer(((self.in_features*3)+self.coref_layer_in), config['ffnn_depth'], config['ffnn_size'], 1)
    if config['fine_grained']:
      self.coref_layer_f = nn.Linear(self.in_features*2, self.in_features, bias=True)

  def get_layer(self, in_features, number_of_layer, hidden_size, output_size):
    layer_dict = OrderedDict()
    layer_no = 0
    for i in range(number_of_layer):
      layer_dict["hidden_layer_{}".format(i)] = nn.Linear(in_features, hidden_size, bias=True)
      layer_dict["relu_{}".format(i)] = nn.ReLU()
      layer_dict["dropout_layer_{}".format(i)] = nn.Dropout(self.config['dropout_prob'])
      layer_no += 1
      in_features = hidden_size
    layer_dict["output_layer"] = nn.Linear(in_features, output_size, bias=True)
    return nn.Sequential(layer_dict)

  def get_dropout(self, dropout_prob, is_training):
    if is_training:
      return dropout_prob
    return 0
  
  def flatten_emb_by_sentence(self, emb, text_len_mask):
    num_sentences = emb.shape[0]
    max_sentence_length = emb.shape[1]

    emb_rank = len(emb.shape)
    if emb_rank  == 2:
      flattened_emb = torch.reshape(emb, [num_sentences * max_sentence_length])
    elif emb_rank == 3:
      flattened_emb = torch.reshape(emb, [num_sentences * max_sentence_length, emb.shape[2]])
    else:
      raise ValueError("Unsupported rank: {}".format(emb_rank))
    if emb_rank == 3:
      return torch.masked_select(flattened_emb, torch.reshape(text_len_mask, [num_sentences * max_sentence_length, 1])).view(-1, emb.shape[2])
    return torch.masked_select(flattened_emb, torch.reshape(text_len_mask, [num_sentences * max_sentence_length])).view(-1)
  
  def get_candidate_labels(self, candidate_starts, candidate_ends, labeled_starts, labeled_ends, labels):
    same_start = torch.eq(labeled_starts.unsqueeze(1), candidate_starts.unsqueeze(0)) # [num_labeled, num_candidates]
    same_end = torch.eq(labeled_ends.unsqueeze(1), candidate_ends.unsqueeze(0)) # [num_labeled, num_candidates]
    same_span = torch.logical_and(same_start, same_end) # [num_labeled, num_candidates]
    candidate_labels = torch.matmul(labels.unsqueeze(0), same_span.int()) # [1, num_candidates]
    candidate_labels = candidate_labels.squeeze(0) # [num_candidates]
    return candidate_labels
  
  def get_masked_mention_word_scores(self, encoded_doc, span_starts, span_ends):
    num_words = encoded_doc.shape[0] # T
    num_c = span_starts.shape[0] # NC torch.tile(torch.arange(num_words).unsqueeze(1), [1, 30])
    doc_range = torch.tile(torch.arange(num_words).unsqueeze(0), [num_c, 1]) # [K, T]
    mention_mask = torch.logical_and(doc_range >= span_starts.unsqueeze(1), doc_range <= span_ends.unsqueeze(1)) #[K, T]
    word_attn = self.mention_word_attn(encoded_doc).squeeze(1)
    mention_word_attn = nn.Softmax(dim=1)(torch.log(mention_mask.long()) + word_attn.unsqueeze(0))
    return mention_word_attn
      
 
  def get_span_emb(self, head_emb, context_outputs, span_starts, span_ends, dropout):
    span_emb_list = []

    span_start_emb = context_outputs[span_starts] # [k, emb]
    span_emb_list.append(span_start_emb)

    span_end_emb = context_outputs[span_ends] # [k, emb]
    span_emb_list.append(span_end_emb)

    span_width = 1 + span_ends - span_starts # [k]

    if self.config['use_features']:
      span_width_index = span_width - 1 # [k]
      span_width_emb = self.span_width_embedding[span_width_index]
      span_width_emb = dropout(span_width_emb)
      span_emb_list.append(span_width_emb)
      
    if self.config['model_heads']:
      mention_word_scores = self.get_masked_mention_word_scores(context_outputs, span_starts, span_ends)
      head_attn_reps = torch.matmul(mention_word_scores, context_outputs) # [K, T]
      span_emb_list.append(head_attn_reps)
    span_emb = torch.cat(span_emb_list, 1)
    return span_emb
  
  def get_mention_scores(self, span_emb, span_starts, span_ends):
    span_scores = self.mention_span_scores(span_emb) # [k, 1]
    if self.config['use_prior']:#use_prior
      span_width_index = span_ends - span_starts # [NC]
      width_scores = self.mention_width_scores(self.span_width_prior_embedding) # [k, 1]
      width_scores = width_scores[span_width_index]
      span_scores += width_scores
    return span_scores

  def get_fast_antecedent_scores(self, top_span_emb, dropout):
    source_top_span_emb = dropout(self.src_projection(top_span_emb)) # [k, emb]
    target_top_span_emb = dropout(top_span_emb) # [k, emb]
    return torch.matmul(source_top_span_emb, target_top_span_emb.T) # [k, k]
  
  def bucket_distance(self, distances):
    """
    Places the given values (designed for distances) into 10 semi-logscale buckets:
    [0, 1, 2, 3, 4, 5-7, 8-15, 16-31, 32-63, 64+].
    """
    logspace_idx = (torch.floor(torch.log(distances)/math.log(2)) + 3).int()
    use_identity = (distances <= 4).int()
    combined_idx = use_identity * distances + (1 - use_identity) * logspace_idx
    return torch.clamp(combined_idx, 0, 9)
  
  def batch_gather(self, emb, indices):
    batch_size = emb.shape[0]
    seqlen = emb.shape[1]
    if len(emb.shape) > 2:
      emb_size = _shape(emb, 2)
    else:
      emb_size = 1
    flattened_emb = torch.reshape(emb, [batch_size * seqlen, emb_size])  # [batch_size * seqlen, emb]
    offset = (torch.arange(batch_size)*seqlen).unsqueeze(1)  # [batch_size, 1]
    gathered = flattened_emb[indices + offset] # [batch_size, num_indices, emb]
    if len(emb.shape) == 2:
      gathered = gathered.squeeze() # [batch_size, num_indices]
    return gathered

  def coarse_to_fine_pruning(self, top_span_emb, top_span_mention_scores, c, dropout):
    k = top_span_emb.shape[0]
    top_span_range = torch.arange(top_span_emb.shape[0]) # [k]
    antecedent_offsets = top_span_range.unsqueeze(1) - top_span_range.unsqueeze(0) # [k, k]
    antecedents_mask = antecedent_offsets >= 1 # [k, k]
    fast_antecedent_scores = top_span_mention_scores.unsqueeze(1) + top_span_mention_scores.unsqueeze(0) # [k, k]
    fast_antecedent_scores += torch.log(antecedents_mask) # [k, k]
    fast_antecedent_scores += self.get_fast_antecedent_scores(top_span_emb, dropout) # [k, k]
    if self.config['use_prior']:
      antecedent_distance_buckets = self.bucket_distance(antecedent_offsets) # [k, c]
      distance_scores = self.distance_scores_linear(dropout(self.antecedent_distance_embedding)) #[10, 1]
      antecedent_distance_scores = distance_scores.squeeze(1)[antecedent_distance_buckets] # [k, c]
      fast_antecedent_scores += antecedent_distance_scores

    _, top_antecedents = fast_antecedent_scores.topk(c, sorted=False) # [k, c]
    top_antecedents_mask = self.batch_gather(antecedents_mask, top_antecedents) # [k, c]
    top_fast_antecedent_scores = self.batch_gather(fast_antecedent_scores, top_antecedents) # [k, c]
    top_antecedent_offsets = self.batch_gather(antecedent_offsets, top_antecedents) # [k, c]
    return top_antecedents, top_antecedents_mask, top_fast_antecedent_scores, top_antecedent_offsets
  
  def get_slow_antecedent_scores(self, top_span_emb, top_antecedents, top_antecedent_emb, top_antecedent_offsets, top_span_speaker_ids, genre_emb, segment_distance=None, dropout=None):
    k = top_span_emb.shape[0]
    c = top_antecedents.shape[1]
    feature_emb_list = []

    if self.config['use_metadata']: #use_metadata
      top_antecedent_speaker_ids = top_span_speaker_ids[top_antecedents.long()] # [k, c]
      same_speaker = torch.eq(top_span_speaker_ids.unsqueeze(1), top_antecedent_speaker_ids) # [k, c]
      speaker_pair_emb = self.coref_layer_speaker_embedding[same_speaker.long()] # [k, c, emb]
      feature_emb_list.append(speaker_pair_emb)

      tiled_genre_emb = torch.tile(genre_emb.unsqueeze(0).unsqueeze(0), [k, c, 1]) # [k, c, emb]
      feature_emb_list.append(tiled_genre_emb)

    if self.config['use_features']: #use feature
      antecedent_distance_buckets = self.bucket_distance(top_antecedent_offsets) # [k, c]
      antecedent_distance_emb = self.coref_layer_antecedent_distance_embedding[antecedent_distance_buckets] # [k, c]
      feature_emb_list.append(antecedent_distance_emb)
    
    if segment_distance is not None:
      segment_distance_emb = self.coref_layer_segment_distance_embedding[segment_distance] # [k, emb]
      feature_emb_list.append(segment_distance_emb)
    
    feature_emb = torch.cat(feature_emb_list, 2) # [k, c, emb]
    feature_emb = dropout(feature_emb) # [k, c, emb]

    target_emb = top_span_emb.unsqueeze(1) # [k, 1, emb]
    similarity_emb = top_antecedent_emb * target_emb # [k, c, emb]
    target_emb = torch.tile(target_emb, [1, c, 1]) # [k, c, emb]

    pair_emb = torch.cat([target_emb, top_antecedent_emb, similarity_emb, feature_emb], 2) # [k, c, emb]

    slow_antecedent_scores = self.coref_layer_slow_antecedent_scores(pair_emb) # [k, 1]
    slow_antecedent_scores = slow_antecedent_scores.squeeze(2) # [k, c]
    return slow_antecedent_scores # [k, c]
  
  def softmax_loss(self, antecedent_scores, antecedent_labels):
    gold_scores = antecedent_scores + torch.log(antecedent_labels.long()) # [k, max_ant + 1]
    marginalized_gold_scores = torch.logsumexp(gold_scores, [1]) # [k]
    log_norm = torch.logsumexp(antecedent_scores, [1]) # [k]
    return log_norm - marginalized_gold_scores # [k]

  def forward(self, input_ids, input_mask, text_len, speaker_ids, genre, is_training, gold_starts, gold_ends, cluster_ids, sentence_map):
    out = self.bert(input_ids=input_ids, attention_mask=input_mask)
    dropout = nn.Dropout(self.get_dropout(self.config['dropout_prob'], is_training), inplace=False)

    num_sentences = out['last_hidden_state'].shape[0]
    max_sentence_length = out['last_hidden_state'].shape[1]
    mention_doc = self.flatten_emb_by_sentence(out['last_hidden_state'], input_mask.bool())
    num_words = mention_doc.shape[0]
    flattened_sentence_indices = sentence_map

    candidate_starts = torch.tile(torch.arange(num_words).unsqueeze(1), [1, 30])
    candidate_ends = candidate_starts + torch.arange(30).unsqueeze(0)
    candidate_start_sentence_indices = flattened_sentence_indices[candidate_starts]
    candidate_end_sentence_indices = flattened_sentence_indices[torch.minimum(candidate_ends, torch.tensor(num_words - 1))]
    candidate_mask = torch.logical_and(candidate_ends < num_words, torch.eq(candidate_start_sentence_indices, candidate_end_sentence_indices))
    flattened_candidate_mask = candidate_mask.view(-1,)
    candidate_starts = torch.masked_select(candidate_starts.view(-1), flattened_candidate_mask) # [num_candidates]
    candidate_ends = torch.masked_select(candidate_ends.view(-1), flattened_candidate_mask) # [num_candidates]
    candidate_sentence_indices = torch.masked_select(candidate_start_sentence_indices.view(-1), flattened_candidate_mask) # [num_candidates]
    candidate_cluster_ids = self.get_candidate_labels(candidate_starts, candidate_ends, gold_starts, gold_ends, cluster_ids)

    candidate_span_emb = self.get_span_emb(mention_doc, mention_doc, candidate_starts, candidate_ends, dropout)
    candidate_mention_scores =  self.get_mention_scores(candidate_span_emb, candidate_starts, candidate_ends)
    
    candidate_mention_scores = candidate_mention_scores.squeeze(1) # [k]

    # beam size
    k = int(np.minimum(3900, np.floor(num_words * self.config["top_span_ratio"])))
    c = int(np.minimum(self.config["max_top_antecedents"], k))
    top_span_indices = util.get_top_span_indices(candidate_mention_scores, candidate_starts, candidate_ends, num_words, k)
    indices = torch.tensor(top_span_indices.numpy()).long().squeeze()
    top_span_starts = candidate_starts[indices] # [k]
    top_span_ends = candidate_ends[indices] # [k]
    top_span_emb = candidate_span_emb[indices] # [k, emb]
    top_span_cluster_ids = candidate_cluster_ids[indices] # [k]
    top_span_mention_scores = candidate_mention_scores[indices]
    genre_emb = self.genre_embedding[genre] # [emb]
    if self.config['use_metadata']: #meta_data
        speaker_ids = self.flatten_emb_by_sentence(speaker_ids, input_mask.bool())
        top_span_speaker_ids = speaker_ids[top_span_starts] # [k]i
    else:
        top_span_speaker_ids = None
    
    top_antecedents, top_antecedents_mask, top_fast_antecedent_scores, top_antecedent_offsets = self.coarse_to_fine_pruning(top_span_emb, top_span_mention_scores, c, dropout)
    num_segs, seg_len = input_ids.shape[0], input_ids.shape[1]
    word_segments = torch.tile(torch.arange(num_segs).unsqueeze(1), [1, seg_len])
    flat_word_segments = torch.masked_select(torch.reshape(word_segments, [-1]), torch.reshape(input_mask, [-1]).bool())
    mention_segments = flat_word_segments[top_span_starts].unsqueeze(1) # [k, 1]
    antecedent_segments = flat_word_segments[top_span_starts[top_antecedents]] #[k, c]
    segment_distance = torch.clamp(mention_segments - antecedent_segments, 0, self.config['max_training_sentences'] - 1) if self.config['use_segment_distance'] else None #[k, c]#segment_distance
    
    dummy_scores = torch.zeros([k, 1])
    if self.config['fine_grained']: #fine_grained
      for i in range(self.config['coref_depth']):#coref_depth
        top_antecedent_emb = top_span_emb[top_antecedents] # [k, c, emb]
        top_antecedent_scores = top_fast_antecedent_scores + self.get_slow_antecedent_scores(top_span_emb, top_antecedents, top_antecedent_emb, top_antecedent_offsets, top_span_speaker_ids.long(), genre_emb, segment_distance, dropout) # [k, c]
        top_antecedent_weights = torch.nn.Softmax()(torch.cat([dummy_scores, top_antecedent_scores], 1)) # [k, c + 1]
        top_antecedent_emb = torch.cat([top_span_emb.unsqueeze(1), top_antecedent_emb], 1) # [k, c + 1, emb]
        attended_span_emb = torch.sum(top_antecedent_weights.unsqueeze(2) * top_antecedent_emb, 1) # [k, emb]
        f = torch.sigmoid(self.coref_layer_f(torch.cat([top_span_emb, attended_span_emb], 1))) # [k, emb]
        top_span_emb = f * attended_span_emb + (1 - f) * top_span_emb # [k, emb]
    else:
      top_antecedent_scores = top_fast_antecedent_scores

    top_antecedent_scores = torch.cat([dummy_scores, top_antecedent_scores], 1) # [k, c + 1]
    top_antecedent_cluster_ids = top_span_cluster_ids[top_antecedents] # [k, c]
    top_antecedent_cluster_ids += torch.log(top_antecedents_mask.long()).int() # [k, c]
    same_cluster_indicator = torch.eq(top_antecedent_cluster_ids, top_span_cluster_ids.unsqueeze(1)) # [k, c]
    non_dummy_indicator = (top_span_cluster_ids > 0).unsqueeze(1) # [k, 1]
    pairwise_labels = torch.logical_and(same_cluster_indicator, non_dummy_indicator) # [k, c]
    dummy_labels = torch.logical_not(torch.all(pairwise_labels, 1, keepdims=True)) # [k, 1]
    top_antecedent_labels = torch.cat([dummy_labels, pairwise_labels], 1) # [k, c + 1]
    loss = self.softmax_loss(top_antecedent_scores, top_antecedent_labels) # [k]
    loss = torch.sum(loss) # []
    return [candidate_starts, candidate_ends, candidate_mention_scores, top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores], loss
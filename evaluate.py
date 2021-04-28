import util
import torch
import metrics
import numpy as np


def get_predicted_antecedents(antecedents, antecedent_scores):
    predicted_antecedents = []
    for i, index in enumerate(np.argmax(antecedent_scores, axis=1) - 1):
      if index < 0:
        predicted_antecedents.append(-1)
      else:
        predicted_antecedents.append(antecedents[i, index])
    return predicted_antecedents

def get_predicted_clusters(top_span_starts, top_span_ends, predicted_antecedents):
    mention_to_predicted = {}
    predicted_clusters = []
    for i, predicted_index in enumerate(predicted_antecedents):
      if predicted_index < 0:
        continue
      assert i > predicted_index, (i, predicted_index)
      predicted_antecedent = (int(top_span_starts[predicted_index]), int(top_span_ends[predicted_index]))
      if predicted_antecedent in mention_to_predicted:
        predicted_cluster = mention_to_predicted[predicted_antecedent]
      else:
        predicted_cluster = len(predicted_clusters)
        predicted_clusters.append([predicted_antecedent])
        mention_to_predicted[predicted_antecedent] = predicted_cluster

      mention = (int(top_span_starts[i]), int(top_span_ends[i]))
      predicted_clusters[predicted_cluster].append(mention)
      mention_to_predicted[mention] = predicted_cluster

    predicted_clusters = [tuple(pc) for pc in predicted_clusters]
    mention_to_predicted = { m:predicted_clusters[i] for m,i in mention_to_predicted.items() }

    return predicted_clusters, mention_to_predicted

def evaluate_coref(top_span_starts, top_span_ends, predicted_antecedents, gold_clusters, evaluator):
    gold_clusters = [tuple(tuple(m) for m in gc) for gc in gold_clusters]
    mention_to_gold = {}
    for gc in gold_clusters:
      for mention in gc:
        mention_to_gold[mention] = gc

    predicted_clusters, mention_to_predicted = get_predicted_clusters(top_span_starts, top_span_ends, predicted_antecedents)
    evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)
    return predicted_clusters

def evaluate(model, config, device):
    eval_dataloader = util.get_dataloader(config, evaluation=True)
    print("Number eval example: {}".format(len(eval_dataloader)))

    coref_predictions = {}
    coref_evaluator = metrics.CorefEvaluator()
    losses = []
    doc_keys = []
    model.eval()

    for i, batch in enumerate(eval_dataloader):
      doc_keys.append(batch['doc_key'])
      (candidate_starts, candidate_ends, candidate_mention_scores, top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores), loss = model.forward(batch['input_ids'].to(device), batch['input_mask'].to(device), batch['text_len'], batch['speaker_ids'].to(device), batch['genre'], False, batch['gold_starts'].to(device), batch['gold_ends'].to(device), batch['cluster_ids'].to(device), batch['sentence_map'].to(device))
      losses.append(loss)
      predicted_antecedents = get_predicted_antecedents(top_antecedents.cpu().detach().numpy(), top_antecedent_scores.cpu().detach().numpy())
      coref_predictions[batch["doc_key"]] = evaluate_coref(top_span_starts.cpu().detach().numpy(), top_span_ends.cpu().detach().numpy(), predicted_antecedents, batch["clusters"], coref_evaluator)
      if i % 10 == 0:
        print("Evaluated {}/{} examples.".format(i + 1, len(eval_dataloader)))
    
    p,r,f = coref_evaluator.get_prf()
    print("Average F1 (py): {:.2f}% on {} docs".format(f * 100, len(doc_keys)))
    print("Average precision (py): {:.2f}%".format(p * 100))
    print("Average recall (py): {:.2f}%".format(r * 100))

    return f
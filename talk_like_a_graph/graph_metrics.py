"""Metrics for seqio tasks over graph data.

This module contains definitions of metric_fns to be used for scoring
graph tasks from nlgraph and graphqa.
"""

from typing import Mapping, Sequence


def yes_no_accuracy(
    targets: Sequence[str], predictions: Sequence[str]
) -> Mapping[str, float]:
  """Assesses the accuracy of LLM outputs on Yes/No tasks.
  
  Targets must contain either the word 'yes' or the word 'no' but not both.
  
  Predictions are binarized by checking for 'yes' or 'no' in the first line.

  Args:
    targets: The expected output strings.
    predictions: The LLM outputs.

  Returns:
     Returns a dict of the following metrics:
      yes_no_accuracy: The % where the target and prediction match.
      yes_no_ambiguous: The % where the prediction contained yes and no
      yes_no_indeterminate: The % where the prediction contained neither yes nor
      no

  Raises:
    ValueError: If a target string contains 'yes' and 'no'
  """
  num_correct = 0
  num_ambiguous = 0
  num_indeterminate = 0
  for target, prediction in zip(targets, predictions):
    normalized_target = target.lower()
    binarized_target = 'yes' in normalized_target
    print(binarized_target)
    if binarized_target and 'no' in normalized_target:
      raise ValueError(f'Ambiguous target string, {target}')
    if not binarized_target and 'no' not in normalized_target:
      raise ValueError(f'Indeterminate target string, {target}')
    normalized_prediction = prediction.splitlines()
    if not normalized_prediction:
      normalized_prediction = ''
    else:
      normalized_prediction = normalized_prediction[0]
    normalized_prediction = normalized_prediction.lower()
    binarized_prediction = 'yes' in normalized_prediction.lower()
    print(normalized_prediction)
    if binarized_prediction and 'no' in normalized_prediction:
      num_ambiguous += 1
      continue
    if not binarized_prediction and 'no' not in normalized_prediction:
      num_indeterminate += 1
      continue
    if binarized_prediction == binarized_target:
      num_correct += 1
  return {
      'yes_no_accuracy': num_correct / len(targets),
      'yes_no_ambiguous': num_ambiguous / len(targets),
      'yes_no_indeterminate': num_indeterminate / len(targets),
  }

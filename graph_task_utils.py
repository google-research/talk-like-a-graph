# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The graph tasks to be tried with LLMs."""

import os
import random

import networkx as nx
import tensorflow as tf
from tensorflow.io import gfile

from google3.third_party.google_research.google_research.graphqa import graph_task
from google3.third_party.tensorflow.core.example import example_pb2
from google3.third_party.tensorflow.core.example import feature_pb2


def create_example_feature(
    key: int,
    question: str,
    answer: str,
    algorithm: str,
    encoding_method: str,
    nnodes: str,
    nedges: str,
) -> example_pb2.Example:
  """Create a tensorflow example from a datapoint."""
  key_feature = feature_pb2.Feature(
      bytes_list=tf.train.BytesList(value=[str(key).encode()])
  )
  question_feature = feature_pb2.Feature(
      bytes_list=tf.train.BytesList(value=[question.encode()])
  )
  answer_feature = feature_pb2.Feature(
      bytes_list=tf.train.BytesList(value=[answer.encode()])
  )
  algorithm_feature = feature_pb2.Feature(
      bytes_list=tf.train.BytesList(value=[algorithm.encode()])
  )
  encoding_method_feature = feature_pb2.Feature(
      bytes_list=tf.train.BytesList(value=[encoding_method.encode()])
  )
  nnodes_feature = feature_pb2.Feature(
      bytes_list=tf.train.BytesList(value=[nnodes.encode()])
  )
  nedges_feature = feature_pb2.Feature(
      bytes_list=tf.train.BytesList(value=[nedges.encode()])
  )
  example_feats = tf.train.Features(
      feature={
          'id': key_feature,
          'question': question_feature,
          'answer': answer_feature,
          'algorithm': algorithm_feature,
          'text_encoding': encoding_method_feature,
          'nnodes': nnodes_feature,
          'nedges': nedges_feature,
      }
  )
  return example_pb2.Example(features=example_feats)


def load_graphs(
    base_path: str,
    algorithm: str,
    split: str,
    max_nnodes: int = 20,
) -> list[nx.Graph]:
  """Load a list of graphs from a given algorithm and split."""
  graphs_path = os.path.join(
      base_path,
      algorithm,
      split,
  )
  loaded_graphs = []
  all_files = gfile.listdir(graphs_path)
  for file in all_files:
    if file.endswith('.graphml'):
      path = os.path.join(graphs_path, file)
      graph = nx.read_graphml(open(path, 'rb'), node_type=int)
      if graph.number_of_nodes() <= max_nnodes:
        loaded_graphs.append(graph)
  return loaded_graphs


def prepare_examples(
    examples_dict: dict[int, dict[str, str | list[int]]],
    encoding_method: str,
) -> list[example_pb2.Example]:
  """Create a list of tf.train.Example from a dict of examples."""
  examples = []
  for key, value in examples_dict.items():
    (
        question,
        answer,
        nnodes,
        nedges,
        algorithm,
    ) = (
        value['question'],
        value['answer'],
        value['nnodes'],
        value['nedges'],
        value['algorithm'],
    )
    examples.append(
        create_example_feature(
            key,
            question,
            answer,
            algorithm,
            encoding_method,
            nnodes,
            nedges,
        )
    )
  return examples


def create_zero_shot_task(
    task: graph_task.GraphTask,
    graphs: list[nx.Graph],
    generator_algorithms: list[str],
    text_encoders: list[str],
    cot: bool = False,
) -> list[example_pb2.Example]:
  """Create a recordio file with zero-shot examples for the task."""
  examples = []
  for encoding_method in text_encoders:
    examples_dict = task.prepare_examples_dict(
        graphs, generator_algorithms, encoding_method
    )
    if cot:
      for key in examples_dict.keys():
        examples_dict[key]['question'] += "Let's think step by step. "
    examples += prepare_examples(examples_dict, encoding_method)
  return examples


def write_examples(examples: list[example_pb2.Example], output_path: str):
  with tf.io.TFRecordWriter(output_path) as file_writer:
    for example in examples:
      file_writer.write(example.SerializeToString())


def prepare_few_shots(
    task: graph_task.GraphTask,
    graphs: list[nx.Graph],
    text_encoders: list[str],
    cot: bool,
) -> dict[str, list[str]]:
  """Create a dict of few-shot examples with their cot for the task."""
  few_shots_examples_dict = {}
  for encoding_method in text_encoders:
    if encoding_method not in few_shots_examples_dict:
      few_shots_examples_dict[(encoding_method)] = []
    for graph in graphs:
      few_shots_examples_dict[(encoding_method)].append(
          task.create_few_shot_example(graph, encoding_method, cot)
      )
  return few_shots_examples_dict


def choose_few_shot_examples(
    few_shots_dict: dict[str, list[str]],
    encoding_method: str,
    k: int = 2,
) -> str:
  """Choose few shot examples for each algorithm."""
  few_shots_str = ''
  for _ in range(k):
    example_list = few_shots_dict[encoding_method]
    few_shots_str += 'Example: ' + random.choice(example_list) + '\n'
  return few_shots_str


def create_few_shot_task(
    task: graph_task.GraphTask,
    graphs: list[nx.Graph],
    generator_algorithms: list[str],
    few_shots_graphs: list[nx.Graph],
    text_encoders: list[str],
    cot: bool,
    bag: bool,
    random_seed: int,
) -> list[example_pb2.Example]:
  """Create a recordio file with few-shot examples for the task."""
  number_of_tokens = {}
  examples = []
  print('prepare few shot task', 'cot', cot, 'bag', bag)
  few_shots_examples_dict = prepare_few_shots(
      task,
      few_shots_graphs,
      text_encoders,
      cot,
  )
  for encoding_method in text_encoders:
    random.seed(random_seed)
    examples_dict = task.prepare_examples_dict(
        graphs, generator_algorithms, encoding_method
    )
    for key in examples_dict.keys():
      few_shots_examples = choose_few_shot_examples(
          few_shots_examples_dict,
          encoding_method,
      )
      examples_dict[key]['question'] = (
          few_shots_examples + 'Example: ' + examples_dict[key]['question']
      )
      if bag:
        examples_dict[key]['question'] = examples_dict[key]['question'].replace(
            '\nQ: ',
            "\nLet's construct the graph with the nodes and edges first.\nQ: ",
        )  # pytype: disable=attribute-error
      if encoding_method not in number_of_tokens:
        number_of_tokens[encoding_method] = []
    examples += prepare_examples(examples_dict, encoding_method)

  return examples

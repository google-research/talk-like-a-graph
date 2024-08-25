"""The graph tasks to be tried with LLMs."""

import os
import random

import networkx as nx
import numpy as np
import seqio
import tensorflow as tf
import tensorflow_gnn as tfgnn

# Google-internal import(s).
# Internal import.
from . import graph_tasks
from tensorflow.core.example import example_pb2
from tensorflow.core.example import feature_pb2


def laplacian_pos_embedding(graph: nx.Graph, units: int = 4) -> nx.Graph:
  """Adds the laplacian positional encoding."""
  m = nx.normalized_laplacian_matrix(
      graph, nodelist=sorted(graph.nodes), weight=None
  ).astype(np.float32)
  u, _, _ = np.linalg.svd(m.todense(), compute_uv=True)
  if units > u.shape[1]:
    u = np.pad(u, ((0, 0), (0, units - u.shape[1])))
  nx.set_node_attributes(
      graph, dict(zip(sorted(graph.nodes), u[:, :units])), name='lpe'
  )
  return graph


def to_tfgnn(graph: nx.Graph, node_ids: list[int]) -> tfgnn.GraphTensor:
  """Convert a given nx graph to a tfgnn graph."""
  if graph.edges(data=True):
    s, t, w = zip(*[
        (s, t, (d['weight'] if d and 'weight' in d else None))
        for s, t, d in graph.edges(data=True)
    ])
  else:
    s, t, w = (), (), ()
  # tfgnn assumes graphs are directed. Adding the rev edges for an undirected
  # graph.
  if not graph.is_directed():
    s, t, w = s + t, t + s, w + w

  graph = laplacian_pos_embedding(graph, units=4)
  features = set(k for n in graph.nodes for k in graph.nodes[n].keys())  # pylint: disable=g-complex-comprehension
  node_features = {
      f: tf.convert_to_tensor([graph.nodes[n][f] for n in graph.nodes])
      for f in features
  }
  # If all edges have a non-trivial weight, then we record the weights.
  if all(w):
    edge_features = {'weights': tf.convert_to_tensor(w, dtype=tf.int32)}
    gt = tfgnn.homogeneous(
        tf.convert_to_tensor(s, dtype=tf.int32),
        tf.convert_to_tensor(t, dtype=tf.int32),
        node_features=node_features,
        edge_features=edge_features,
    )
  else:
    gt = tfgnn.homogeneous(
        tf.convert_to_tensor(s, dtype=tf.int32),
        tf.convert_to_tensor(t, dtype=tf.int32),
        node_features=node_features,
    )

  if not node_ids:
    # No node is mentioned in the task description.
    return gt
  node_sets = {
      **gt.node_sets,
      '_readout': tfgnn.NodeSet.from_fields(sizes=[1]),
  }
  if len(node_ids) == 1:
    # Tasks requiring only one node id e.g., computing node degree.
    edge_sets = {
        **gt.edge_sets,
        '_readout/node': tfgnn.EdgeSet.from_fields(
            sizes=[1],
            adjacency=tfgnn.Adjacency.from_indices(
                source=('nodes', node_ids),
                target=('_readout', [0]),
            ),
        ),
    }
  elif len(node_ids) == 2:
    # Tasks requiring two nodes e.g., shortest path from one node to the other.
    edge_sets = {
        **gt.edge_sets,
        '_readout/source': tfgnn.EdgeSet.from_fields(
            sizes=[1],
            adjacency=tfgnn.Adjacency.from_indices(
                source=('nodes', node_ids[:1]),
                target=('_readout', [0]),
            ),
        ),
        '_readout/target': tfgnn.EdgeSet.from_fields(
            sizes=[1],
            adjacency=tfgnn.Adjacency.from_indices(
                source=('nodes', node_ids[1:]),
                target=('_readout', [0]),
            ),
        ),
    }
  else:
    # Raising an error if more than two nodes are mentiones.
    raise ValueError(f'Invalid number of integers: {len(node_ids)}')

  return tfgnn.GraphTensor.from_pieces(
      context=gt.context, node_sets=node_sets, edge_sets=edge_sets
  )


def create_example_feature(
    key: int,
    question: str,
    answer: str,
    algorithm: str,
    encoding_method: str,
    nnodes: str,
    nedges: str,
    task_description: str,
    graph: nx.Graph,
    node_ids: list[int],
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
  task_description_feature = feature_pb2.Feature(
      bytes_list=tf.train.BytesList(value=[task_description.encode()])
  )
  gt = to_tfgnn(graph, node_ids)
  graph_feature = feature_pb2.Feature(
      bytes_list=tf.train.BytesList(
          value=[tfgnn.write_example(gt).SerializeToString()]
      )
  )
  directed_feature = feature_pb2.Feature(
      bytes_list=tf.train.BytesList(value=[str(graph.is_directed()).encode()])
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
          'task_description': task_description_feature,
          'graph': graph_feature,
          'directed': directed_feature,
      }
  )
  return example_pb2.Example(features=example_feats)


def load_graphs(
    base_path: str,
    algorithm: str,
    split: str,
    direction: str,
    max_nnodes: int = 20,
) -> list[nx.Graph]:
  """Load a list of graphs from a given algorithm and split."""
  graphs_path = os.path.join(
      base_path,
      direction,
      algorithm,
      split,
  )
  loaded_graphs = []
  all_files = gfile.ListDir(graphs_path)
  for file in all_files:
    if file.endswith('.graphml'):
      path = os.path.join(graphs_path, file)
      graph = nx.read_graphml(os.Open(path, 'rb'), node_type=int)
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
        task_description,
        graph,
        algorithm,
        node_ids,
    ) = (
        value['question'],
        value['answer'],
        value['nnodes'],
        value['nedges'],
        value['task_description'],
        value['graph'],
        value['algorithm'],
        value['node_ids'],
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
            task_description,
            graph,
            node_ids,
        )
    )
  return examples


def create_zero_shot_task(
    task: graph_tasks.GraphTask,
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
  with recordio.RecordWriter(output_path) as output_file:
    for example in examples:
      output_file.WriteRecord(example.SerializeToString())


def prepare_few_shots(
    task: graph_tasks.GraphTask,
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
    task: graph_tasks.GraphTask,
    graphs: list[nx.Graph],
    generator_algorithms: list[str],
    few_shots_graphs: list[nx.Graph],
    text_encoders: list[str],
    cot: bool,
    bag: bool,
    random_seed: int,
) -> list[example_pb2.Example]:
  """Create a recordio file with few-shot examples for the task."""
  # LINT.IfChange
  vocab_path = None
  # LINT.ThenChange(//research/graph/llm/graphqa/copy.bara.sky)
  # Loading the palm tokenizer to calculate number of tokens in the sequence.
  sp_vocab = seqio.SentencePieceVocabulary(vocab_path)
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
      number_of_tokens[encoding_method].append(
          len(sp_vocab.encode(examples_dict[key]['question']))
      )
    examples += prepare_examples(examples_dict, encoding_method)

  # Printing maximum number of tokens in the sequence.
  for key, value in number_of_tokens.items():
    print(key, np.max(value))

  return examples

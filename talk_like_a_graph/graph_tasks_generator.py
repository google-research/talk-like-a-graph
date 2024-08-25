r"""The graph tasks to be tried with LLMs..

This code loads graphs and creates graph tasks and output them as tf examples in
a recordio file in the task directory provided.

# Placeholder for Google-internal comments.
"""

from collections.abc import Sequence
import os
import random

from absl import app
from absl import flags
import networkx as nx
import numpy as np

from . import graph_tasks
from . import graph_tasks_utils as utils

_TASK_DIR = flags.DEFINE_string(
    'task_dir', None, 'The directory to write tasks.', required=True
)
_GRAPHS_DIR = flags.DEFINE_string(
    'graphs_dir', None, 'The directory containing the graphs.', required=True
)
_RANDOM_SEED = flags.DEFINE_integer(
    'random_seed',
    None,
    'The random seed to use for task generation.',
    required=True,
)


def zero_shot(
    task: graph_tasks.GraphTask,
    graphs: list[nx.Graph],
    algorithms: list[str],
    text_encoders: list[str],
    cot: bool,
    random_seed: int,
    split: str,
) -> None:
  """Creating zero-shot or zero-cot examples for the given task.

  Args:
    task: the corresponding graph task.
    graphs: the list of graphs to use for the task.
    algorithms: the algorithm used to generate the graphs.
    text_encoders: the encoders to use in the tasks.
    cot: whether to apply cot or not.
    random_seed: the random seed to use in the process.
    split: whether we are creating a train or test split.
  """
  random.seed(random_seed)
  zero_shot_examples = utils.create_zero_shot_task(
      task, graphs, algorithms, text_encoders, cot=cot
  )

  file_name = task.name + ('_zero_cot_' if cot else '_zero_shot_')

  file_name += split + '.recordio'
  utils.write_examples(
      zero_shot_examples,
      os.path.join(_TASK_DIR.value, file_name),
  )


def few_shot(
    task: graph_tasks.GraphTask,
    graphs: list[nx.Graph],
    few_shot_graphs: list[nx.Graph],
    algorithms: list[str],
    text_encoders: list[str],
    cot: bool,
    bag: bool,
    random_seed: int,
) -> None:
  """Creating few-shot, cot, or cot-bag examples for the given task.

  Args:
    task: the corresponding graph task.
    graphs: the list of graphs to use for the task.
    few_shot_graphs: the list of graphs to generate few shot examples for.
    algorithms: the algorithm used to generate the graphs.
    text_encoders: the encoders to use in the tasks.
    cot: whether to apply cot or not.
    bag: whether to apply build-a-graph method or not.
    random_seed: the random seed to use in the process.
  """
  random.seed(random_seed)
  few_shot_examples = utils.create_few_shot_task(
      task,
      graphs,
      algorithms,
      few_shot_graphs,
      text_encoders,
      cot=cot,
      bag=bag,
      random_seed=random_seed,
  )
  file_name = task.name
  if cot and bag:
    file_name += '_few_shot_cot_bag_test.recordio'
  elif cot:
    file_name += '_few_shot_cot_test.recordio'
  else:
    file_name += '_few_shot_test.recordio'

  utils.write_examples(
      few_shot_examples,
      os.path.join(_TASK_DIR.value, file_name),
  )


def generate_random_sbm_graph(random_state: np.random.RandomState):
  # Sampling a small number as the probability of the two nodes in different
  # communities being connected.
  small_number = random.uniform(0, 0.05)
  # Sampling a large number as probability of the nodes in one community
  # being connected.
  large_number = random.uniform(0.6, 0.8)
  number_of_nodes = random.choice(np.arange(5, 20))
  sizes = [number_of_nodes // 2, number_of_nodes // 2]
  probs = [[large_number, small_number], [small_number, large_number]]
  return nx.stochastic_block_model(sizes, probs, seed=random_state)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  algorithms = ['er']
  directions = ['undirected']
  text_encoders = ['adjacency']

  # Loading the graphs.
  graphs = []
  generator_algorithms = []
  for algorithm in algorithms:
    for direction in directions:
      loaded_graphs = utils.load_graphs(
          _GRAPHS_DIR.value,
          algorithm,
          'train',
          direction,
      )
      graphs += loaded_graphs
      generator_algorithms += [algorithm] * len(loaded_graphs)

  # Defining a task on the graphs
  task = graph_tasks.ShortestPath()

  if isinstance(task, graph_tasks.NodeClassification):
    # The node classification task requires SBM graphs. As it's not possible to
    # write graphs with data (e.g., blocks data as in SBM graphs), we regenerate
    # graphs.

    random_state = np.random.RandomState(_RANDOM_SEED.value)
    print('Generating sbm graphs')
    graphs = [
        generate_random_sbm_graph(random_state) for _ in range(len(graphs))
    ]

  zero_shot(
      task,
      graphs,
      generator_algorithms,
      text_encoders,
      cot=False,
      random_seed=_RANDOM_SEED.value,
      split='test',
  )
  zero_shot(
      task,
      graphs,
      generator_algorithms,
      text_encoders,
      cot=True,
      random_seed=_RANDOM_SEED.value,
      split='test',
  )

  # Loading few-shot graphs.
  few_shot_graphs = []
  for algorithm in algorithms:
    for direction in directions:
      few_shot_graphs += utils.load_graphs(
          _GRAPHS_DIR.value,
          algorithm,
          'train',
          direction,
      )

  if isinstance(task, graph_tasks.NodeClassification):
    # The node classification task requires SBM graphs. As it's not possible to
    # write graphs with data (e.g., blocks data as in SBM graphs), we regenerate
    # graphs.
    random_state = np.random.RandomState(_RANDOM_SEED.value + 1)
    print('Generating few shot sbm graphs')
    few_shot_graphs = [
        generate_random_sbm_graph(random_state)
        for _ in range(len(few_shot_graphs))
    ]

  few_shot(
      task,
      graphs,
      few_shot_graphs,
      generator_algorithms,
      text_encoders,
      cot=False,
      bag=False,
      random_seed=_RANDOM_SEED.value,
  )

  few_shot(
      task,
      graphs,
      few_shot_graphs,
      generator_algorithms,
      text_encoders,
      cot=True,
      bag=False,
      random_seed=_RANDOM_SEED.value,
  )

  few_shot(
      task,
      graphs,
      few_shot_graphs,
      generator_algorithms,
      text_encoders,
      cot=True,
      bag=True,
      random_seed=_RANDOM_SEED.value,
  )


if __name__ == '__main__':
  app.run(main)

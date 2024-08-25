r"""Random graph generation.

This code generates random graph using different algorithms.

# Placeholder for Google-internal comments.
"""

from collections.abc import Sequence
import os

from absl import app
from absl import flags
import networkx as nx

# Internal import.
from . import graph_generators

_ALGORITHM = flags.DEFINE_string(
    "algorithm",
    None,
    "The graph generating algorithm to use.",
    required=True,
)
_NUMBER_OF_GRAPHS = flags.DEFINE_integer(
    "number_of_graphs",
    None,
    "The number of graphs to generate.",
    required=True,
)
_DIRECTED = flags.DEFINE_bool(
    "directed", False, "Whether to generate directed graphs."
)
_OUTPUT_PATH = flags.DEFINE_string(
    "output_path", None, "The output path to write the graphs.", required=True
)
_SPLIT = flags.DEFINE_string(
    "split", None, "The dataset split to generate.", required=True
)
_MIN_SPARSITY = flags.DEFINE_float("min_sparsity", 0.0, "The minimum sparsity.")
_MAX_SPARSITY = flags.DEFINE_float("max_sparsity", 1.0, "The maximum sparsity.")


def write_graphs(graphs: list[nx.Graph], output_dir: str) -> None:
  if not os.path.isdir(output_dir):
    os.makedirs(output_dir)
  for ind, graph in enumerate(graphs):
    nx.write_graphml(
        graph,
        os.Open(
            os.path.join(output_dir, str(ind) + ".graphml"),
            "wb",
        ),
    )


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  if _SPLIT.value == "train":
    random_seed = 9876
  elif _SPLIT.value == "test":
    random_seed = 1234
  elif _SPLIT.value == "validation":
    random_seed = 5432
  else:
    raise NotImplementedError()

  generated_graphs = graph_generators.generate_graphs(
      number_of_graphs=_NUMBER_OF_GRAPHS.value,
      algorithm=_ALGORITHM.value,
      directed=_DIRECTED.value,
      random_seed=random_seed,
      er_min_sparsity=_MIN_SPARSITY.value,
      er_max_sparsity=_MAX_SPARSITY.value,
  )
  write_graphs(
      graphs=generated_graphs,
      output_dir=os.path.join(
          _OUTPUT_PATH.value,
          "directed" if _DIRECTED.value else "undirected",
          _ALGORITHM.value,
          _SPLIT.value,
      ),
  )


if __name__ == "__main__":
  app.run(main)

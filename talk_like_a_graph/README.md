# Using Large Language Models to Solve Graph Problems

This repository contains the code to generate graph reasoning problems with
different graph generator algorithms and graph encoding methods, as well as
different prompting techniques.

The graph tasks are `edge existence`, `node degree`, `node count`, `edge count`,
`connected nodes`, `disconnected nodes`, `cycle check`, `reachability`,
`shortest path`, `maximum flow`, `node classification`, and `triangle counting`.

The datasets used here are proposed in our paper:
[Talk like a Graph: Encoding Graphs for Large Language Models](https://arxiv.org/abs/2310.04560).

### Generating graphs

```sh
./graphqa/graph_generator.sh
```

### Generating files for tasks

```sh
./graphqa/task_generator.sh
```

## Contact us

For questions or comments about the implementation, please contact
baharef@google.com.

## Cite us

If you use this package for published work, please cite the following:

```
@inproceedigs{fatemi2024talk,
  title={Talk like a Graph: Encoding Graphs for Large Language Models},
  author={Bahare Fatemi and Jonathan Halcrow and Bryan Perozzi},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2024}
}
```

## Disclaimer

This is not an official Google product.

# Placeholder for internal data notes.
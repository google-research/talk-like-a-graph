from absl.testing import parameterized
from . import graph_generators
from absl.testing import absltest


class GraphGenerationTest(absltest.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='er_undirected_1',
          algorithm='er',
          directed=False,
          k=1,
      ),
      dict(
          testcase_name='er_directed_1',
          algorithm='er',
          directed=True,
          k=1,
      ),
      dict(
          testcase_name='ba_undirected_5',
          algorithm='ba',
          directed=False,
          k=5,
      ),
      dict(
          testcase_name='ba_directed_5',
          algorithm='ba',
          directed=True,
          k=5,
      ),
  )
  def test_number_of_graphs(self, algorithm, directed, k):
    generated_graph = graph_generators.generate_graphs(k, algorithm, directed)
    self.assertLen(generated_graph, k)

  @parameterized.named_parameters(
      dict(
          testcase_name='er_undirected',
          algorithm='er',
          directed=False,
      ),
      dict(
          testcase_name='er_directed',
          algorithm='er',
          directed=True,
      ),
      dict(
          testcase_name='ba_undirected',
          algorithm='ba',
          directed=False,
      ),
      dict(
          testcase_name='ba_directed',
          algorithm='ba',
          directed=True,
      ),
      dict(
          testcase_name='sbm_undirected',
          algorithm='sbm',
          directed=False,
      ),
      dict(
          testcase_name='sbm_directed',
          algorithm='sbm',
          directed=True,
      ),
      dict(
          testcase_name='sfn_undirected',
          algorithm='sfn',
          directed=False,
      ),
      dict(
          testcase_name='sfn_directed',
          algorithm='sfn',
          directed=True,
      ),
      dict(
          testcase_name='complete_undirected',
          algorithm='complete',
          directed=False,
      ),
      dict(
          testcase_name='complete_directed',
          algorithm='complete',
          directed=True,
      ),
      dict(
          testcase_name='star_undirected',
          algorithm='star',
          directed=False,
      ),
      dict(
          testcase_name='star_directed',
          algorithm='star',
          directed=True,
      ),
      dict(
          testcase_name='path_undirected',
          algorithm='path',
          directed=False,
      ),
      dict(
          testcase_name='path_directed',
          algorithm='path',
          directed=True,
      ),
  )
  def test_directions(self, algorithm, directed):
    generated_graph = graph_generators.generate_graphs(1, algorithm, directed)
    self.assertEqual(generated_graph[0].is_directed(), directed)


if __name__ == '__main__':
  googletest.main()

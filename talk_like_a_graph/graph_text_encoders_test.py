"""Testing for graph_text_encoders.py."""

from absl.testing import parameterized
import networkx as nx

from . import graph_text_encoders
from absl.testing import absltest

_G = nx.Graph()
_G.add_node(0)
_G.add_node(1)
_G.add_node(2)
_G.add_node(3)
_G.add_edge(0, 1)
_G.add_edge(1, 2)
_G.add_edge(2, 3)
_G.add_edge(3, 0)


class GraphTextEncodersTest(absltest.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='adjacency_integer',
          encoding_method='adjacency',
          expected_result=(
              'In an undirected graph, (i,j) means that node i and node j are'
              ' connected with an undirected edge. G describes a graph among'
              ' nodes 0, 1, 2, and 3.\nThe edges in G are: (0, 1) (0, 3) (1, 2)'
              ' (2, 3).\n'
          ),
      ),
      dict(
          testcase_name='incident_integer',
          encoding_method='incident',
          expected_result=(
              'G describes a graph among nodes 0, 1, 2, and 3.\nIn this'
              ' graph:\nNode 0 is connected to nodes 1, 3.\nNode 1 is connected'
              ' to nodes 0, 2.\nNode 2 is connected to nodes 1, 3.\nNode 3 is'
              ' connected to nodes 2, 0.\n'
          ),
      ),
      dict(
          testcase_name='friendship_per_line_popular',
          encoding_method='friendship',
          expected_result=(
              'G describes a friendship graph among nodes James, Robert, John,'
              ' and Michael.\nWe have the following edges in G:\nJames and'
              ' Robert are friends.\nJames and Michael are friends.\nRobert and'
              ' John are friends.\nJohn and Michael are friends.\n'
          ),
      ),
      dict(
          testcase_name='social_network_politician',
          encoding_method='politician',
          expected_result=(
              'G describes a social network graph among nodes Barack, Jimmy,'
              ' Arnold, and Bernie.\nWe have the following edges in'
              ' G:\nBarack and Jimmy are connected.\nBarack and Bernie are'
              ' connected.\nJimmy and Arnold are connected.\nArnold and'
              ' Bernie are connected.\n'
          ),
      ),
  )
  def test_encoders(self, encoding_method, expected_result):
    self.assertEqual(
        graph_text_encoders.encode_graph(_G, encoding_method),
        expected_result,
    )


if __name__ == '__main__':
  googletest.main()

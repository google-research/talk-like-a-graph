"""Library for encoding graphs in text."""

import networkx as nx

from . import name_dictionaries


def create_node_string(name_dict, nnodes: int) -> str:
  node_string = ""
  sorted_keys = list(sorted(name_dict.keys()))
  for i in sorted_keys[: nnodes - 1]:
    node_string += name_dict[i] + ", "
  node_string += "and " + name_dict[sorted_keys[nnodes - 1]]
  return node_string


def nx_encoder(graph: nx.Graph, _: dict[int, str], edge_type="id") -> str:
  """Encoding a graph as entries of an adjacency matrix."""
  if graph.is_directed():
    output = (
        "In a directed graph, (s,p,o) means that there is an edge from node s"
        " to node o of type p. "
    )
  else:
    output = (
        "In an undirected graph, (s,p,o) means that node s and node o are"
        " connected with an undirected edge of type p. "
    )

  name_dict = {x: str(x) for x in graph.nodes()}

  nodes_string = create_node_string(name_dict, nnodes=len(graph.nodes()))
  output += "G describes a graph among nodes %s.\n" % nodes_string
  if graph.edges():
    output += "The edges in G are: "
  for i, j in graph.edges():
    edge_type = graph.get_edge_data(i, j)[edge_type]
    if edge_type is None:
      edge_type = "linked"
    output += "(%s, %s, %s) " % (name_dict[i], edge_type, name_dict[j])
  return output.strip() + ".\n"


def adjacency_encoder(graph: nx.Graph, name_dict: dict[int, str]) -> str:
  """Encoding a graph as entries of an adjacency matrix."""
  if graph.is_directed():
    output = (
        "In a directed graph, (i,j) means that there is an edge from node i to"
        " node j. "
    )
  else:
    output = (
        "In an undirected graph, (i,j) means that node i and node j are"
        " connected with an undirected edge. "
    )
  nodes_string = create_node_string(name_dict, len(graph.nodes()))
  output += "G describes a graph among nodes %s.\n" % nodes_string
  if graph.edges():
    output += "The edges in G are: "
  for i, j in graph.edges():
    output += "(%s, %s) " % (name_dict[i], name_dict[j])
  return output.strip() + ".\n"


def friendship_encoder(graph: nx.Graph, name_dict: dict[int, str]) -> str:
  """Encoding a graph as a friendship graph."""
  if graph.is_directed():
    raise ValueError("Friendship encoder is not defined for directed graphs.")
  nodes_string = create_node_string(name_dict, len(graph.nodes()))
  output = (
      "G describes a friendship graph among nodes %s.\n" % nodes_string.strip()
  )
  if graph.edges():
    output += "We have the following edges in G:\n"
  for i, j in graph.edges():
    output += "%s and %s are friends.\n" % (name_dict[i], name_dict[j])
  return output


def coauthorship_encoder(graph: nx.Graph, name_dict: dict[int, str]) -> str:
  """Encoding a graph as a coauthorship graph."""
  if graph.is_directed():
    raise ValueError("Coauthorship encoder is not defined for directed graphs.")
  nodes_string = create_node_string(name_dict, len(graph.nodes()))
  output = (
      "G describes a coauthorship graph among nodes %s.\n"
      % nodes_string.strip()
  )
  if graph.edges():
    output += "In this coauthorship graph:\n"
  for i, j in graph.edges():
    output += "%s and %s wrote a paper together.\n" % (
        name_dict[i],
        name_dict[j],
    )
  return output.strip() + ".\n"


def incident_encoder(graph: nx.Graph, name_dict: dict[int, str]) -> str:
  """Encoding a graph with its incident lists."""
  nodes_string = create_node_string(name_dict, len(graph.nodes()))
  output = "G describes a graph among nodes %s.\n" % nodes_string
  if graph.edges():
    output += "In this graph:\n"
  for source_node in graph.nodes():
    target_nodes = graph.neighbors(source_node)
    target_nodes_str = ""
    nedges = 0
    for target_node in target_nodes:
      target_nodes_str += name_dict[target_node] + ", "
      nedges += 1
    if nedges > 1:
      output += "Node %s is connected to nodes %s.\n" % (
          source_node,
          target_nodes_str[:-2],
      )
    elif nedges == 1:
      output += "Node %d is connected to node %s.\n" % (
          source_node,
          target_nodes_str[:-2],
      )
  return output


def social_network_encoder(graph: nx.Graph, name_dict: dict[int, str]) -> str:
  """Encoding a graph as a social network graph."""
  if graph.is_directed():
    raise ValueError(
        "Social network encoder is not defined for directed graphs."
    )
  nodes_string = create_node_string(name_dict, len(graph.nodes()))
  output = (
      "G describes a social network graph among nodes %s.\n"
      % nodes_string.strip()
  )
  if graph.edges():
    output += "We have the following edges in G:\n"
  for i, j in graph.edges():
    output += "%s and %s are connected.\n" % (name_dict[i], name_dict[j])
  return output


def expert_encoder(graph: nx.Graph, name_dict: dict[int, str]) -> str:
  nodes_string = create_node_string(name_dict, len(graph.nodes()))
  output = (
      "You are a graph analyst and you have been given a graph G among nodes"
      " %s.\n"
      % nodes_string.strip()
  )
  output += "G has the following undirected edges:\n" if graph.edges() else ""
  for i, j in graph.edges():
    output += "%s -> %s\n" % (name_dict[i], name_dict[j])
  return output


def nodes_to_text(graph, encoding_type):
  """Get dictionary converting node ids to text."""
  if encoding_type == "integer":
    return name_dictionaries.create_name_dict(graph, "integer", nnodes=1000)
  elif encoding_type == "popular":
    return name_dictionaries.create_name_dict(graph, "popular")
  elif encoding_type == "alphabet":
    return name_dictionaries.create_name_dict(graph, "alphabet")
  elif encoding_type == "got":
    return name_dictionaries.create_name_dict(graph, "got")
  elif encoding_type == "south_park":
    return name_dictionaries.create_name_dict(graph, "south_park")
  elif encoding_type == "politician":
    return name_dictionaries.create_name_dict(graph, "politician")
  elif encoding_type == "random":
    return name_dictionaries.create_name_dict(
        graph, "random_integer", nnodes=1000
    )
  elif encoding_type == "nx_node_name":
    return name_dictionaries.create_name_dict(graph, "nx_node_name")
  else:
    raise ValueError("Unknown encoding type: %s" % encoding_type)


def get_tlag_node_encoder(graph, encoder_name):
  """Find the node encoder used in the 'Talk Like a Graph' paper."""
  if encoder_name == "adjacency":
    return nodes_to_text(graph, "integer")
  elif encoder_name == "incident":
    return nodes_to_text(graph, "integer")
  elif encoder_name == "friendship":
    return nodes_to_text(graph, "popular")
  elif encoder_name == "south_park":
    return nodes_to_text(graph, "south_park")
  elif encoder_name == "got":
    return nodes_to_text(graph, "got")
  elif encoder_name == "politician":
    return nodes_to_text(graph, "politician")
  elif encoder_name == "social_network":
    return nodes_to_text(graph, "popular")
  elif encoder_name == "expert":
    return nodes_to_text(graph, "expert")
  elif encoder_name == "coauthorship":
    return nodes_to_text(graph, "popular")
  elif encoder_name == "random":
    return nodes_to_text(graph, "random")
  elif encoder_name == "nx_node_name":
    return nodes_to_text(graph, "nx_node_name")
  else:
    raise ValueError("Unknown graph encoder strategy: %s" % encoder_name)


# A dictionary from edge encoder name to the corresponding function.
EDGE_ENCODER_FN = {
    "adjacency": adjacency_encoder,
    "incident": incident_encoder,
    "friendship": friendship_encoder,
    "south_park": friendship_encoder,
    "got": friendship_encoder,
    "politician": social_network_encoder,
    "social_network": social_network_encoder,
    "expert": expert_encoder,
    "coauthorship": coauthorship_encoder,
    "random": adjacency_encoder,
    "nx_edge_encoder": nx_encoder,
}


def with_ids(graph: nx.Graph, node_encoder: str) -> nx.Graph:
  nx.set_node_attributes(graph, nodes_to_text(graph, node_encoder), name="id")
  return graph


def encode_graph(
    graph: nx.Graph, graph_encoder=None, node_encoder=None, edge_encoder=None
) -> str:
  r"""Encodes a graph as text.

  This relies on choosing:
     a node_encoder and an edge_encoder:
     or
     a graph_encoder (a predefined pair of node and edge encoding strategies).

  Note that graph_encoders may assume that the graph has some properties
  (e.g. integer keys).

  Example usage:
  .. code-block:: python
  ```
  # Use a predefined graph encoder from the paper.
  >>> G = nx.karate_club_graph()
  >>> encode_graph(G, graph_encoder="adjacency")
  'In an undirected graph, (i,j) means that node i and node j are
  connected
  with an undirected edge. G describes a graph among nodes 0, 1, 2, 3, 4, 5,
  6,
  7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
  27, 28, 29, 30, 31, 32, and 33.\nThe edges in G are: (0, 1) (0, 2) (0, 3)
  ...'

  # Use the node's name in the graph as the node identifier.
  >>> G = nx.les_miserables_graph()
  >>> encode_graph(G, node_encoder="nx_node_name", edge_encoder="friendship")
  'G describes a friendship graph among nodes Anzelma, Babet, Bahorel,
  Bamatabois, BaronessT, Blacheville, Bossuet, Boulatruelle, Brevet, ...
  We have the following edges in G:
  Napoleon and Myriel are friends. Myriel and MlleBaptistine are friends...'

  # Use the `id` feature from the edges to describe the edge type.
  >>> G = nx.karate_club_graph()
  >>> encode_graph(G, node_encoder="nx_node_name", edge_encoder="nx_edge_id")
  'In an undirected graph, (s,p,o) means that node s and node o are connected
  with an undirected edge of type p. G describes a graph among nodes 0, 1, 2, 3,
  4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
  25, 26, 27, 28, 29, 30, 31, 32, and 33.
  The edges in G are: (0, linked, 1) (0, linked, 2) (0, linked, 3) ...'
  ```

  Args:
    graph: the graph to be encoded.
    graph_encoder: the name of the graph encoder to use.
    node_encoder: the name of the node encoder to use.
    edge_encoder: the name of the edge encoder to use.

  Returns:
    The encoded graph as a string.
  """

  # Check that only one of graph_encoder or (node_encoder, edge_encoder) is set.
  if graph_encoder and (node_encoder or edge_encoder):
    raise ValueError(
        "Only one of graph_encoder or (node_encoder, edge_encoder) can be set."
    )

  if graph_encoder:
    if isinstance(graph_encoder, str):
      node_encoder_dict = get_tlag_node_encoder(graph, graph_encoder)
      return EDGE_ENCODER_FN[graph_encoder](graph, node_encoder_dict)
    else:
      return graph_encoder(graph)

  else:
    node_encoder_dict = nodes_to_text(graph, node_encoder)
    return EDGE_ENCODER_FN[edge_encoder](graph, node_encoder_dict)

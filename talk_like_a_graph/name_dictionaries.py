"""Creates a dictionary mapping integers to node names."""

import random

_RANDOM_SEED = 1234
random.seed(_RANDOM_SEED)

_INTEGER_NAMES = [str(x) for x in range(10000)]

_POPULAR_NAMES = [
    "James",
    "Robert",
    "John",
    "Michael",
    "David",
    "Mary",
    "Patricia",
    "Jennifer",
    "Linda",
    "Elizabeth",
    "William",
    "Richard",
    "Joseph",
    "Thomas",
    "Christopher",
    "Barbara",
    "Susan",
    "Jessica",
    "Sarah",
    "Karen",
    "Daniel",
    "Lisa",
    "Matthew",
    "Nancy",
    "Anthony",
    "Betty",
    "Mark",
    "Margaret",
    "Donald",
    "Sandra",
    "Steven",
    "Ashley",
    "Paul",
    "Kimberly",
    "Andrew",
    "Emily",
    "Joshua",
    "Donna",
    "Kenneth",
    "Michelle",
    "Kevin",
    "Carol",
    "Brian",
    "Amanda",
    "George",
    "Melissa",
    "Edward",
    "Deborah",
    "Ronald",
    "Stephanie",
    "Timothy",
    "Rebecca",
    "Jason",
    "Sharon",
    "Jeffrey",
    "Laura",
    "Ryan",
    "Cynthia",
    "Jacob",
    "Dorothy",
    "Gary",
    "Olivia",
    "Nicholas",
    "Emma",
    "Eric",
    "Sophia",
    "Jonathan",
    "Ava",
    "Stephen",
    "Isabella",
    "Scott",
    "Mia",
    "Justin",
    "Abigail",
    "Brandon",
    "Madison",
    "Frank",
    "Chloe",
    "Benjamin",
    "Victoria",
    "Samuel",
    "Lauren",
    "Gregory",
    "Hannah",
    "Alexander",
    "Grace",
    "Frank",
    "Alexis",
    "Raymond",
    "Alice",
    "Patrick",
    "Samantha",
    "Jack",
    "Natalie",
    "Dennis",
    "Anna",
    "Jerry",
    "Taylor",
    "Tyler",
    "Kayla",
    "Henry",
    "Hailey",
    "Douglas",
    "Jasmine",
    "Peter",
    "Nicole",
    "Adam",
    "Amy",
    "Nathan",
    "Christina",
    "Zachary",
    "Andrea",
    "Jose",
    "Leah",
    "Walter",
    "Angelina",
    "Harold",
    "Valerie",
    "Kyle",
    "Veronica",
    "Ethan",
    "Carl",
    "Arthur",
    "Roger",
    "Noah",
]


_SOUTH_PARK_NAMES = [
    "Eric",
    "Kenny",
    "Kyle",
    "Stan",
    "Tolkien",
    "Heidi",
    "Bebe",
    "Liane",
    "Sharon",
    "Linda",
    "Gerald",
    "Veronica",
    "Michael",
    "Jimbo",
    "Herbert",
    "Malcolm",
    "Gary",
    "Steve",
    "Chris",
    "Wendy",
]

_GOT_NAMES = [
    "Ned",
    "Cat",
    "Daenerys",
    "Jon",
    "Bran",
    "Sansa",
    "Arya",
    "Cersei",
    "Jaime",
    "Petyr",
    "Robert",
    "Jorah",
    "Viserys",
    "Joffrey",
    "Maester",
    "Theon",
    "Rodrik",
    "Lysa",
    "Stannis",
    "Osha",
]


_POLITICIAN_NAMES = [
    "Barack",
    "Jimmy",
    "Arnold",
    "Bernie",
    "Bill",
    "Kamala",
    "Hillary",
    "Elizabeth",
    "John",
    "Ben",
    "Joe",
    "Alexandria",
    "George",
    "Nancy",
    "Pete",
    "Madeleine",
    "Elijah",
    "Gabrielle",
    "Al",
]


_ALPHABET_NAMES = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
    "AA",
    "BB",
    "CC",
    "DD",
    "EE",
    "FF",
    "GG",
    "HH",
    "II",
    "JJ",
    "KK",
    "LL",
    "MM",
    "NN",
    "OO",
    "PP",
    "QQ",
    "RR",
    "SS",
    "TT",
    "UU",
    "VV",
    "WW",
    "XX",
    "YY",
    "ZZ",
]


def create_name_dict(graph, name: str, nnodes: int = 20) -> dict[int, str]:
  """The runner function to map integers to node names.

  Args:
    graph: the graph to be encoded.
    name: name of the approach for mapping.
    nnodes: optionally provide nnodes in the graph to be encoded.

  Returns:
    A dictionary from integers to strings.
  """
  if name == "alphabet":
    names_list = _ALPHABET_NAMES
  elif name == "integer":
    names_list = _INTEGER_NAMES
  elif name == "random_integer":
    names_list = []
    for _ in range(nnodes):
      names_list.append(str(random.randint(0, 1000000)))
  elif name == "popular":
    names_list = _POPULAR_NAMES
  elif name == "south_park":
    names_list = _SOUTH_PARK_NAMES
  elif name == "got":
    names_list = _GOT_NAMES
  elif name == "politician":
    names_list = _POLITICIAN_NAMES
  elif name == "nx_node_name":
    return {x: str(x) for x in graph.nodes()}
  else:
    raise ValueError(f"Unknown approach: {name}")
  name_dict = {}
  for ind, value in enumerate(names_list):
    name_dict[ind] = value

  return name_dict

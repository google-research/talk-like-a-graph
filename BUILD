load("//devtools/python/blaze:pytype.bzl", "pytype_strict_binary", "pytype_strict_library")

package(
    default_applicable_licenses = ["//third_party/google_research/google_research:license"],
    default_visibility = ["//visibility:public"],
)

licenses(["notice"])

pytype_strict_library(
    name = "graph_generator_utils",
    srcs = ["graph_generator_utils.py"],
    deps = [
        "//third_party/py/networkx",
        "//third_party/py/numpy",
    ],
)

pytype_strict_binary(
    name = "graph_generator",
    srcs = ["graph_generator.py"],
    deps = [
        ":graph_generator_utils",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/networkx",
        "//third_party/py/tensorflow",
    ],
)

pytype_strict_library(
    name = "name_dictionaries",
    srcs = ["name_dictionaries.py"],
    visibility = ["//visibility:private"],
    deps = [
    ],
)

pytype_strict_library(
    name = "graph_text_encoder",
    srcs = ["graph_text_encoder.py"],
    deps = [
        ":name_dictionaries",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/networkx",
    ],
)

pytype_strict_library(
    name = "graph_task",
    srcs = ["graph_task.py"],
    visibility = ["//visibility:private"],
    deps = [
        ":graph_text_encoder",
        "//third_party/py/absl:app",
        "//third_party/py/networkx",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
        "//third_party/tensorflow/core:protos_all_py_pb2",
    ],
)

pytype_strict_library(
    name = "graph_task_utils",
    srcs = ["graph_task_utils.py"],
    visibility = ["//visibility:private"],
    deps = [
        ":graph_task",
        ":graph_text_encoder",
        "//third_party/py/absl:app",
        "//third_party/py/networkx",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
        "//third_party/tensorflow/core:protos_all_py_pb2",
    ],
)

pytype_strict_binary(
    name = "graph_task_generator",
    srcs = ["graph_task_generator.py"],
    visibility = ["//visibility:private"],
    deps = [
        ":graph_task",
        ":graph_task_utils",
        ":graph_text_encoder",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/networkx",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
        "//third_party/tensorflow/core:protos_all_py_pb2",
    ],
)

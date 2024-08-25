from . import graph_metrics
from absl.testing import absltest


class GraphTasksTest(absltest.TestCase):

  def test_yes_no_correct(self):
    result = graph_metrics.yes_no_accuracy(
        targets=["Yes", "The answer is yes.", "No", "The answer is no.", "No"],
        predictions=[
            "yes",
            "Yes\nNo",
            "No\nYes",
            "That's gonna be no from me.",
            "yes",
        ],
    )
    self.assertEqual(result["yes_no_ambiguous"], 0)
    self.assertEqual(result["yes_no_indeterminate"], 0)
    self.assertAlmostEqual(result["yes_no_accuracy"], 0.8)

  def test_yes_no_ambiguous(self):
    result = graph_metrics.yes_no_accuracy(
        targets=["Yes", "The answer is yes.", "No", "The answer is no.", "No"],
        predictions=[
            "yes",
            "Yes No",
            "No Yes",
            "That's gonna be no from me.",
            "yes",
        ],
    )
    self.assertEqual(result["yes_no_ambiguous"], 0.4)
    self.assertEqual(result["yes_no_indeterminate"], 0)
    self.assertAlmostEqual(result["yes_no_accuracy"], 0.4)

  def test_yes_no_indeterminate(self):
    result = graph_metrics.yes_no_accuracy(
        targets=["Yes", "The answer is yes.", "No", "The answer is no.", "No"],
        predictions=[
            "yes",
            "\n No",
            "",
            "That's gonna be no from me.",
            "yes",
        ],
    )
    self.assertEqual(result["yes_no_ambiguous"], 0)
    self.assertEqual(result["yes_no_indeterminate"], 0.4)
    self.assertAlmostEqual(result["yes_no_accuracy"], 0.4)

  def test_yes_no_accuracy_raises_on_ambiguous_target(self):
    with self.assertRaises(ValueError):
      graph_metrics.yes_no_accuracy(
          targets=["Yes but maybe no"],
          predictions=["yes"],
      )

  def test_yes_no_accuracy_raises_on_indeterminate_target(self):
    with self.assertRaises(ValueError):
      graph_metrics.yes_no_accuracy(
          targets=["Hmm?"],
          predictions=["yes"],
      )


if __name__ == "__main__":
  googletest.main()

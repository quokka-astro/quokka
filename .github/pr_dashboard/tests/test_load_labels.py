import pathlib
import tempfile
import unittest

from pr_dashboard import load_labels
from pr_dashboard import pr_labels

CONFIGURED_LABELS = load_labels.load_configured_labels(pathlib.Path(__file__).parents[1] / "labels.toml")
SCOPE_LABEL_TEXT = '[[scope.label]]\nvalue = "hydro"\ncolor = "4ADD08"\ndescription = "Hydro"\npath_patterns = ["src/hydro/*"]\n'


def make_priority_label_text(
    *,
    value: str,
    is_default: bool = False,
) -> str:
    if is_default:
        default_line = "default = true\n"
    else:
        default_line = ""
    return f'[[priority.label]]\nvalue = "{value}"\ncolor = "FBCA04"\ndescription = "Priority"\n{default_line}'


LOW_PRIORITY_LABEL_TEXT = make_priority_label_text(value="low")
NORMAL_PRIORITY_LABEL_TEXT = make_priority_label_text(value="normal")
DEFAULT_PRIORITY_LABEL_TEXT = make_priority_label_text(
    value="normal",
    is_default=True,
)
HIGH_PRIORITY_LABEL_TEXT = make_priority_label_text(value="high")
VALID_PRIORITY_LABELS_TEXT = LOW_PRIORITY_LABEL_TEXT + DEFAULT_PRIORITY_LABEL_TEXT + HIGH_PRIORITY_LABEL_TEXT


def make_config_text(
    *,
    scope_label_text: str = SCOPE_LABEL_TEXT,
    priority_label_text: str = VALID_PRIORITY_LABELS_TEXT,
) -> str:
    return (
        '[scope]\nprefix = "scope/"\n'
        '[scope.not_mapped_label]\nvalue = "not_mapped"\ncolor = "B60205"\ndescription = "Not mapped"\n'
        f"{scope_label_text}"
        '[size]\nprefix = "size/"\n'
        '[[size.label]]\nvalue = "XS"\ncolor = "00FF00"\ndescription = "Small"\nmin_num_lines = 0\n'
        '[priority]\nprefix = "priority/"\n'
        f"{priority_label_text}"
    )


def load_config_text(
    config_text: str,
) -> pr_labels.ConfiguredLabels:
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = pathlib.Path(temp_dir) / "labels.toml"
        config_path.write_text(config_text)
        return load_labels.load_configured_labels(config_path)


class TestLoadLabels_Categories(unittest.TestCase):

    def test_categories_are_split(
        self,
    ):
        self.assertEqual(len(CONFIGURED_LABELS.scope_labels.labels), 15)
        self.assertEqual(len(CONFIGURED_LABELS.size_labels.labels), 5)
        self.assertEqual(len(CONFIGURED_LABELS.priority_labels.labels), 3)
        self.assertEqual(len(CONFIGURED_LABELS.all_labels()), 24)
        for priority_label in CONFIGURED_LABELS.priority_labels.labels:
            self.assertIsInstance(priority_label, pr_labels.PriorityLabel)
        self.assertIsInstance(CONFIGURED_LABELS.scope_labels.not_mapped_label, pr_labels.NotMappedLabel)

    def test_label_names_join_prefix_and_value(
        self,
    ):
        hydro_label = CONFIGURED_LABELS.scope_labels.labels[0]
        self.assertEqual(hydro_label.value, "hydro")
        self.assertEqual(hydro_label.label_name, "scope/hydro")
        self.assertEqual(CONFIGURED_LABELS.scope_labels.not_mapped_label.label_name, "scope/not_mapped")
        self.assertEqual(CONFIGURED_LABELS.size_labels.labels[0].label_name, "size/XS")

    def test_categories_check_their_prefix(
        self,
    ):
        self.assertTrue(CONFIGURED_LABELS.scope_labels.check_has_prefix("scope/hydro"))
        self.assertFalse(CONFIGURED_LABELS.scope_labels.check_has_prefix("size/XS"))
        self.assertTrue(CONFIGURED_LABELS.priority_labels.check_has_prefix("priority/urgent"))

    def test_default_priority_is_normal(
        self,
    ):
        default_label = CONFIGURED_LABELS.priority_labels.default_label
        self.assertEqual(default_label.label_name, "priority/normal")
        self.assertTrue(default_label.is_default)

    def test_priority_label_lookup(
        self,
    ):
        self.assertEqual(CONFIGURED_LABELS.priority_labels.get_label("high").label_name, "priority/high")
        self.assertEqual(CONFIGURED_LABELS.priority_labels.get_label("High").label_name, "priority/high")
        self.assertIsNone(CONFIGURED_LABELS.priority_labels.get_label("urgent"))

    def test_minimal_config_loads(
        self,
    ):
        configured_labels = load_config_text(make_config_text())
        self.assertEqual(configured_labels.priority_labels.default_label.label_name, "priority/normal")
        self.assertFalse(configured_labels.priority_labels.labels[0].is_default)


class TestLoadLabels_Errors(unittest.TestCase):

    def test_scope_label_without_path_patterns_fails(
        self,
    ):
        config_text = make_config_text(
            scope_label_text='[[scope.label]]\nvalue = "hydro"\ncolor = "4ADD08"\ndescription = "Hydro"\n',
        )
        with self.assertRaises(KeyError):
            load_config_text(config_text)

    def test_category_without_prefix_fails(
        self,
    ):
        config_text = make_config_text().replace('[size]\nprefix = "size/"\n', "")
        with self.assertRaises(KeyError):
            load_config_text(config_text)

    def test_no_default_priority_fails(
        self,
    ):
        config_text = make_config_text(
            priority_label_text=LOW_PRIORITY_LABEL_TEXT + NORMAL_PRIORITY_LABEL_TEXT +
            HIGH_PRIORITY_LABEL_TEXT,
        )
        with self.assertRaisesRegex(ValueError, "got 0"):
            load_config_text(config_text)

    def test_two_default_priorities_fail(
        self,
    ):
        default_high_label_text = make_priority_label_text(
            value="high",
            is_default=True,
        )
        config_text = make_config_text(
            priority_label_text=LOW_PRIORITY_LABEL_TEXT + DEFAULT_PRIORITY_LABEL_TEXT +
            default_high_label_text,
        )
        with self.assertRaisesRegex(ValueError, "got 2"):
            load_config_text(config_text)


if __name__ == "__main__":
    unittest.main()

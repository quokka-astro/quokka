import pathlib
import tomllib

from pr_dashboard import pr_labels


def _load_scope_labels(
    scope_data: dict,
) -> pr_labels.ScopeLabels:
    prefix = scope_data["prefix"]
    labels = tuple(
        pr_labels.ScopeLabel(
            prefix=prefix,
            value=_label_data["value"],
            color=_label_data["color"],
            description=_label_data["description"],
            path_patterns=tuple(_label_data["path_patterns"]),
        ) for _label_data in scope_data["label"]
    )
    not_mapped_label = pr_labels.NotMappedLabel(
        prefix=prefix,
        value=scope_data["not_mapped_label"]["value"],
        color=scope_data["not_mapped_label"]["color"],
        description=scope_data["not_mapped_label"]["description"],
    )
    return pr_labels.ScopeLabels(
        prefix=prefix,
        labels=labels,
        not_mapped_label=not_mapped_label,
        unlabelled_path_patterns=tuple(scope_data["unlabelled_path_patterns"]),
    )


def _load_size_labels(
    size_data: dict,
) -> pr_labels.SizeLabels:
    prefix = size_data["prefix"]
    labels = tuple(
        pr_labels.SizeLabel(
            prefix=prefix,
            value=_label_data["value"],
            color=_label_data["color"],
            description=_label_data["description"],
            min_num_lines=_label_data["min_num_lines"],
        ) for _label_data in size_data["label"]
    )
    return pr_labels.SizeLabels(
        prefix=prefix,
        labels=labels,
    )


def _load_priority_labels(
    priority_data: dict,
) -> pr_labels.PriorityLabels:
    prefix = priority_data["prefix"]
    labels = tuple(
        pr_labels.PriorityLabel(
            prefix=prefix,
            value=_label_data["value"],
            color=_label_data["color"],
            description=_label_data["description"],
        ) for _label_data in priority_data["label"]
    )
    priority_labels = pr_labels.PriorityLabels(
        prefix=prefix,
        labels=labels,
        default_value=priority_data["default_value"],
    )
    default_value_is_a_label = (priority_labels.get_label(priority_labels.default_value) is not None)
    if default_value_is_a_label:
        raise ValueError(
            f"the default priority `{priority_labels.default_value}` must not also be a priority label.",
        )
    return priority_labels


def load_configured_labels(
    config_path: pathlib.Path,
) -> pr_labels.ConfiguredLabels:
    with open(config_path, "rb") as config_file:
        config_data = tomllib.load(config_file)
    scope_labels = _load_scope_labels(config_data["scope"])
    size_labels = _load_size_labels(config_data["size"])
    priority_labels = _load_priority_labels(config_data["priority"])
    return pr_labels.ConfiguredLabels(
        scope_labels=scope_labels,
        size_labels=size_labels,
        priority_labels=priority_labels,
    )

import dataclasses
import fnmatch

from pr_dashboard import pr_context

##
## === LABELS
##


@dataclasses.dataclass(frozen=True)
class Label:
    prefix: str
    value: str
    color: str
    description: str

    @property
    def label_name(
        self,
    ) -> str:
        return self.prefix + self.value


@dataclasses.dataclass(frozen=True)
class ScopeLabel(Label):
    path_patterns: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class NotMappedLabel(Label):
    pass


@dataclasses.dataclass(frozen=True)
class SizeLabel(Label):
    min_num_lines: int


@dataclasses.dataclass(frozen=True)
class PriorityLabel(Label):
    is_default: bool


##
## === LABEL CATEGORIES
##


@dataclasses.dataclass(frozen=True)
class LabelCategory:
    prefix: str

    def check_has_prefix(
        self,
        label_name: str,
    ) -> bool:
        return label_name.startswith(self.prefix)


@dataclasses.dataclass(frozen=True)
class ScopeLabels(LabelCategory):
    labels: tuple[ScopeLabel, ...]
    not_mapped_label: NotMappedLabel


@dataclasses.dataclass(frozen=True)
class SizeLabels(LabelCategory):
    labels: tuple[SizeLabel, ...]


@dataclasses.dataclass(frozen=True)
class PriorityLabels(LabelCategory):
    labels: tuple[PriorityLabel, ...]
    default_label: PriorityLabel

    def get_label(
        self,
        value: str,
    ) -> PriorityLabel | None:
        matching_labels = [_label for _label in self.labels if _label.value.lower() == value.lower()]
        if len(matching_labels) == 0:
            return None
        else:
            return matching_labels[0]


@dataclasses.dataclass(frozen=True)
class ConfiguredLabels:
    scope_labels: ScopeLabels
    size_labels: SizeLabels
    priority_labels: PriorityLabels

    def all_labels(
        self,
    ) -> tuple[Label, ...]:
        return (
            self.scope_labels.labels + (self.scope_labels.not_mapped_label, ) + self.size_labels.labels +
            self.priority_labels.labels
        )


##
## === LABEL COMPUTATION
##


@dataclasses.dataclass(frozen=True)
class _LabelChanges:
    label_names_to_add: tuple[str, ...]
    label_names_to_remove: tuple[str, ...]


def _path_matches_any(
    *,
    file_path: str,
    patterns: tuple[str, ...],
) -> bool:
    return any(fnmatch.fnmatchcase(file_path, _pattern) for _pattern in patterns)


def _compute_scope_label_names(
    *,
    configured_labels: ConfiguredLabels,
    edited_file_paths: tuple[str, ...],
) -> set[str]:
    scope_label_names = set()
    for file_path in edited_file_paths:
        matched_label_names = {
            _scope_label.label_name
            for _scope_label in configured_labels.scope_labels.labels
            if _path_matches_any(
                file_path=file_path,
                patterns=_scope_label.path_patterns,
            )
        }
        if len(matched_label_names) == 0:
            scope_label_names.add(configured_labels.scope_labels.not_mapped_label.label_name)
        else:
            scope_label_names.update(matched_label_names)
    return scope_label_names


def _compute_size_label_name(
    *,
    configured_labels: ConfiguredLabels,
    num_lines_changed: int,
) -> str:
    increasing_size_labels = sorted(
        configured_labels.size_labels.labels,
        key=lambda _size_label: _size_label.min_num_lines,
    )
    size_label_name = increasing_size_labels[0].label_name
    for next_size_label in increasing_size_labels:
        if num_lines_changed >= next_size_label.min_num_lines:
            size_label_name = next_size_label.label_name
    return size_label_name


def _check_label_is_stale(
    *,
    configured_labels: ConfiguredLabels,
    computed_label_names: set[str],
    label_name: str,
) -> bool:
    label_is_managed_by_tool = (
        configured_labels.scope_labels.check_has_prefix(label_name)
        or configured_labels.size_labels.check_has_prefix(label_name)
    )
    return label_is_managed_by_tool and (label_name not in computed_label_names)


def _compute_label_changes(
    *,
    configured_labels: ConfiguredLabels,
    current_label_names: tuple[str, ...],
    edited_file_paths: tuple[str, ...],
    num_lines_changed: int,
) -> _LabelChanges:
    computed_label_names = _compute_scope_label_names(
        configured_labels=configured_labels,
        edited_file_paths=edited_file_paths,
    )
    size_label_name = _compute_size_label_name(
        configured_labels=configured_labels,
        num_lines_changed=num_lines_changed,
    )
    computed_label_names.add(size_label_name)
    priority_is_set = any(
        configured_labels.priority_labels.check_has_prefix(_label_name) for _label_name in current_label_names
    )
    if not priority_is_set:
        computed_label_names.add(configured_labels.priority_labels.default_label.label_name)
    label_names_to_add = sorted(computed_label_names - set(current_label_names))
    label_names_to_remove = sorted(
        _label_name for _label_name in current_label_names if _check_label_is_stale(
            configured_labels=configured_labels,
            computed_label_names=computed_label_names,
            label_name=_label_name,
        )
    )
    return _LabelChanges(
        label_names_to_add=tuple(label_names_to_add),
        label_names_to_remove=tuple(label_names_to_remove),
    )


##
## === UPDATE PLAN
##


def plan_update_from_pr(
    *,
    configured_labels: ConfiguredLabels,
    pull_request: pr_context.PullRequest,
) -> pr_context.UpdatePlan:
    label_changes = _compute_label_changes(
        configured_labels=configured_labels,
        current_label_names=pull_request.label_names,
        edited_file_paths=pull_request.edited_file_paths,
        num_lines_changed=pull_request.num_lines_changed,
    )
    return pr_context.UpdatePlan(
        label_names_to_add=label_changes.label_names_to_add,
        label_names_to_remove=label_changes.label_names_to_remove,
    )

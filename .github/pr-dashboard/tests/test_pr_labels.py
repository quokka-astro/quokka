import dataclasses
import pathlib
import subprocess
import unittest

from pr_dashboard import load_labels
from pr_dashboard import pr_context
from pr_dashboard import pr_labels

CONFIGURED_LABELS = load_labels.load_configured_labels(pathlib.Path(__file__).parents[1] / "labels.toml")


def make_configured_labels_with_levels(
    relevance_levels: dict[str, int],
) -> pr_labels.ConfiguredLabels:
    scope_labels = tuple(
        dataclasses.replace(
            _scope_label,
            relevance_level=relevance_levels.get(_scope_label.value, 1),
        ) for _scope_label in CONFIGURED_LABELS.scope_labels.labels
    )
    configured_scope_labels = dataclasses.replace(
        CONFIGURED_LABELS.scope_labels,
        labels=scope_labels,
    )
    return dataclasses.replace(
        CONFIGURED_LABELS,
        scope_labels=configured_scope_labels,
    )


## every scope on level 1, so the scope tests check which paths map to which scope, whatever the levels in labels.toml
ALL_SCOPES_SHOWN_LABELS = make_configured_labels_with_levels({})


def get_scope_label_names(
    edited_file_paths: tuple[str, ...],
) -> set[str]:
    return pr_labels._compute_scope_label_names(
        configured_labels=ALL_SCOPES_SHOWN_LABELS,
        edited_file_paths=edited_file_paths,
    )


class TestPrLabels_Scope(unittest.TestCase):

    def test_hydro_file(
        self,
    ):
        scope_label_names = get_scope_label_names(("src/hydro/HLLC.hpp", ))
        self.assertEqual(scope_label_names, {"scope/hydro"})

    def test_mhd_file_inside_hydro_gets_both(
        self,
    ):
        scope_label_names = get_scope_label_names(("src/hydro/HLLD.hpp", ))
        self.assertEqual(scope_label_names, {"scope/hydro", "scope/MHD"})

    def test_problem_directories(
        self,
    ):
        edited_file_paths = (
            "src/problems/MHDBlast/test_mhd_blast.cpp",
            "src/problems/RadDust/test_rad_dust.cpp",
            "src/problems/ParticleSink/test_particle_sink.cpp",
        )
        scope_label_names = get_scope_label_names(edited_file_paths)
        expected_label_names = {
            "scope/MHD",
            "scope/radiation",
            "scope/dust",
            "scope/particles",
            "scope/tests",
        }
        self.assertEqual(scope_label_names, expected_label_names)

    def test_inputs_follow_their_problem(
        self,
    ):
        edited_file_paths = ("inputs/MHDBlast.toml", "inputs/alfven_wave_linear_regression.toml")
        scope_label_names = get_scope_label_names(edited_file_paths)
        self.assertEqual(scope_label_names, {"scope/MHD", "scope/tests"})

    def test_problem_with_no_physics_area_is_tests(
        self,
    ):
        edited_file_paths = ("src/problems/SN/testSN.cpp", "inputs/SN.toml", "extern/Toro/e1rpex.ini")
        scope_label_names = get_scope_label_names(edited_file_paths)
        self.assertEqual(scope_label_names, {"scope/tests"})

    def test_unknown_file_is_not_mapped(
        self,
    ):
        edited_file_paths = ("src/newmodule/thing.hpp", "src/hydro/HLLC.hpp")
        scope_label_names = get_scope_label_names(edited_file_paths)
        self.assertEqual(scope_label_names, {"scope/undefined", "scope/hydro"})

    def test_unlabelled_file_gets_no_scope(
        self,
    ):
        scope_labels = dataclasses.replace(
            CONFIGURED_LABELS.scope_labels,
            unlabelled_path_patterns=("src/newmodule/*", ),
        )
        configured_labels = dataclasses.replace(
            CONFIGURED_LABELS,
            scope_labels=scope_labels,
        )
        scope_label_names = pr_labels._compute_scope_label_names(
            configured_labels=configured_labels,
            edited_file_paths=("src/newmodule/thing.hpp", "src/hydro/HLLC.hpp"),
        )
        self.assertEqual(scope_label_names, {"scope/hydro"})

    def test_build_files_and_libraries(
        self,
    ):
        edited_file_paths = ("src/CMakeLists.txt", "src/problems/ProblemHelpers.cmake", "extern/amrex")
        scope_label_names = get_scope_label_names(edited_file_paths)
        self.assertEqual(scope_label_names, {"scope/build", "scope/tests"})

    def test_problem_registry_is_tests(
        self,
    ):
        scope_label_names = get_scope_label_names(("src/problems/CMakeLists.txt", ))
        self.assertEqual(scope_label_names, {"scope/tests"})

    def test_scripts_are_tools(
        self,
    ):
        scope_label_names = get_scope_label_names(("scripts/python/prune_checkpoints.py", ))
        self.assertEqual(scope_label_names, {"scope/tools"})

    def test_ci_and_documentation(
        self,
    ):
        edited_file_paths = (".github/workflows/cmake.yml", "docs/index.md", "README.md")
        scope_label_names = get_scope_label_names(edited_file_paths)
        self.assertEqual(scope_label_names, {"scope/CI", "scope/documentation"})

    ## keeps the whole codebase accounted for, so the scope labels always show what a PR changes;
    ## a new path needs a scope, or a pattern in `unlabelled_path_patterns` to leave it out on purpose
    def test_every_repository_file_has_a_scope(
        self,
    ):
        repo_root = pathlib.Path(__file__).parents[3]
        completed_process = subprocess.run(
            ["git", "ls-files"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        repository_file_paths = completed_process.stdout.splitlines()
        not_mapped_label_name = CONFIGURED_LABELS.scope_labels.not_mapped_label.label_name
        not_mapped_paths = [
            _file_path for _file_path in repository_file_paths
            if not_mapped_label_name in get_scope_label_names((_file_path, ))
        ]
        self.assertEqual(
            not_mapped_paths,
            [],
            "Add these paths to a scope, or to unlabelled_path_patterns, in .github/pr-dashboard/labels.toml",
        )


def get_scope_label_names_with_levels(
    *,
    relevance_levels: dict[str, int],
    edited_file_paths: tuple[str, ...],
) -> set[str]:
    configured_labels = make_configured_labels_with_levels(relevance_levels)
    return pr_labels._compute_scope_label_names(
        configured_labels=configured_labels,
        edited_file_paths=edited_file_paths,
    )


## physics on level 1, and tests and documentation on level 2
TWO_RELEVANCE_LEVELS = {"tests": 2, "documentation": 2}


class TestPrLabels_RelevanceLevels(unittest.TestCase):

    def test_physics_hides_tests_and_documentation(
        self,
    ):
        scope_label_names = get_scope_label_names_with_levels(
            relevance_levels=TWO_RELEVANCE_LEVELS,
            edited_file_paths=("src/hydro/HLLD.hpp", "src/problems/MHDBlast/test_mhd_blast.cpp", "docs/index.md"),
        )
        self.assertEqual(scope_label_names, {"scope/hydro", "scope/MHD"})

    def test_tests_alone_still_show(
        self,
    ):
        scope_label_names = get_scope_label_names_with_levels(
            relevance_levels=TWO_RELEVANCE_LEVELS,
            edited_file_paths=("src/problems/SN/testSN.cpp", "docs/index.md"),
        )
        self.assertEqual(scope_label_names, {"scope/tests", "scope/documentation"})

    def test_level_one_scopes_all_show(
        self,
    ):
        scope_label_names = get_scope_label_names_with_levels(
            relevance_levels=TWO_RELEVANCE_LEVELS,
            edited_file_paths=("src/radiation/radiation_system.hpp", "src/particles/particle_types.hpp"),
        )
        self.assertEqual(scope_label_names, {"scope/radiation", "scope/particles"})

    def test_unmapped_file_shows_at_any_level(
        self,
    ):
        scope_label_names = get_scope_label_names_with_levels(
            relevance_levels=TWO_RELEVANCE_LEVELS,
            edited_file_paths=("src/newmodule/thing.hpp", "src/hydro/HLLC.hpp", "docs/index.md"),
        )
        self.assertEqual(scope_label_names, {"scope/undefined", "scope/hydro"})

    def test_only_unmapped_file_gives_only_undefined(
        self,
    ):
        scope_label_names = get_scope_label_names_with_levels(
            relevance_levels=TWO_RELEVANCE_LEVELS,
            edited_file_paths=("src/newmodule/thing.hpp", ),
        )
        self.assertEqual(scope_label_names, {"scope/undefined"})

    def test_hidden_scope_label_is_removed_from_pr(
        self,
    ):
        configured_labels = make_configured_labels_with_levels(TWO_RELEVANCE_LEVELS)
        label_changes = pr_labels._compute_label_changes(
            configured_labels=configured_labels,
            current_label_names=("scope/MHD", "scope/tests", "size/M"),
            edited_file_paths=("src/problems/MHDBlast/test_mhd_blast.cpp", ),
            num_lines_changed=40,
        )
        self.assertEqual(label_changes.label_names_to_add, ())
        self.assertEqual(label_changes.label_names_to_remove, ("scope/tests", ))


class TestPrLabels_Size(unittest.TestCase):

    def test_boundaries(
        self,
    ):
        expected_sizes = {
            0: "size/XS",
            9: "size/XS",
            10: "size/S",
            29: "size/S",
            30: "size/M",
            99: "size/M",
            100: "size/L",
            499: "size/L",
            500: "size/XL",
            50000: "size/XL",
        }
        for num_lines_changed, expected_size in expected_sizes.items():
            with self.subTest(num_lines_changed=num_lines_changed):
                size_label_name = pr_labels._compute_size_label_name(
                    configured_labels=CONFIGURED_LABELS,
                    num_lines_changed=num_lines_changed,
                )
                self.assertEqual(size_label_name, expected_size)


class TestPrLabels_Changes(unittest.TestCase):

    def test_new_pr_gets_scope_and_size_but_no_priority(
        self,
    ):
        label_changes = pr_labels._compute_label_changes(
            configured_labels=CONFIGURED_LABELS,
            current_label_names=(),
            edited_file_paths=("src/io/plotfile.cpp", ),
            num_lines_changed=40,
        )
        self.assertEqual(label_changes.label_names_to_add, ("scope/IO", "size/M"))
        self.assertEqual(label_changes.label_names_to_remove, ())

    def test_stale_scope_and_size_are_replaced_and_others_kept(
        self,
    ):
        label_changes = pr_labels._compute_label_changes(
            configured_labels=CONFIGURED_LABELS,
            current_label_names=("scope/radiation", "size/XS", "priority/high", "code-audit"),
            edited_file_paths=("src/io/plotfile.cpp", ),
            num_lines_changed=40,
        )
        self.assertEqual(label_changes.label_names_to_add, ("scope/IO", "size/M"))
        self.assertEqual(label_changes.label_names_to_remove, ("scope/radiation", "size/XS"))

    def test_up_to_date_pr_needs_no_changes(
        self,
    ):
        label_changes = pr_labels._compute_label_changes(
            configured_labels=CONFIGURED_LABELS,
            current_label_names=("scope/IO", "size/M", "priority/low"),
            edited_file_paths=("src/io/plotfile.cpp", ),
            num_lines_changed=40,
        )
        self.assertEqual(label_changes.label_names_to_add, ())
        self.assertEqual(label_changes.label_names_to_remove, ())


class TestPrLabels_UpdatePlan(unittest.TestCase):

    def test_changes_only_labels(
        self,
    ):
        pull_request = pr_context.PullRequest(
            pr_number=1,
            author_user_name="pr-author",
            assignee_user_names=(),
            label_names=("scope/radiation", ),
            num_lines_changed=40,
            edited_file_paths=("src/io/plotfile.cpp", ),
        )
        update_plan_from_pr = pr_labels.plan_update_from_pr(
            configured_labels=CONFIGURED_LABELS,
            pull_request=pull_request,
        )
        expected_update_plan = pr_context.UpdatePlan(
            label_names_to_add=("scope/IO", "size/M"),
            label_names_to_remove=("scope/radiation", ),
        )
        self.assertEqual(update_plan_from_pr, expected_update_plan)


if __name__ == "__main__":
    unittest.main()

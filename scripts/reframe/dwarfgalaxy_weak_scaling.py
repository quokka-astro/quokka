"""ReFrame weak-scaling test for the AgoraGalaxy problem.

This mirrors the manual weak scaling workflow under ``scaling_tests`` by
compiling the ``agora_galaxy`` executable and running it across multiple node
counts while adjusting the mesh size and timestep budget."""

import os
import re
from pathlib import Path

import reframe as rfm
import reframe.utility.sanity as sn
from reframe.core.builtins import parameter, performance_function, run_after, run_before


def _detect_repo_root() -> Path:
    """Infer the Quokka repository root based on common layouts."""
    env_root = os.environ.get('QUOKKA_SOURCE_ROOT')
    if env_root:
        return Path(env_root).resolve()

    here = Path(__file__).resolve().parent
    for candidate in (here, here.parent, here.parent.parent):
        if (candidate / 'scaling_tests').is_dir():
            return candidate

    return here


REPO_ROOT = _detect_repo_root()
TEMPLATES_DIR = REPO_ROOT / 'scaling_tests' / 'templates'
INPUT_TEMPLATE = TEMPLATES_DIR / 'DwarfGalaxy_scaling.in'

TASKS_PER_NODE = 8
CPUS_PER_TASK = 7
GPUS_PER_NODE = 8

WEAK_SCALES = [
    {
        'label': 'n2_256',
        'nodes': 2,
        'cells_per_dim': 256,
        'max_timesteps': 100,
        'prev_max_timesteps': 0,
        'time_limit': '0:15:00',
    },
    {
        'label': 'n16_512',
        'nodes': 16,
        'cells_per_dim': 512,
        'max_timesteps': 200,
        'prev_max_timesteps': 100,
        'time_limit': '0:20:00',
    },
    {
        'label': 'n128_1024',
        'nodes': 128,
        'cells_per_dim': 1024,
        'max_timesteps': 300,
        'prev_max_timesteps': 200,
        'time_limit': '0:30:00',
    },
    {
        'label': 'n1024_2048',
        'nodes': 1024,
        'cells_per_dim': 2048,
        'max_timesteps': 400,
        'prev_max_timesteps': 300,
        'time_limit': '0:45:00',
    },
]


@rfm.simple_test
class DwarfGalaxyWeakScalingTest(rfm.RegressionTest):
    scale = parameter(WEAK_SCALES, fmt=lambda cfg: cfg['label'])

    def __init__(self):
        if not INPUT_TEMPLATE.is_file():
            self.skip(f'Missing template input file: {INPUT_TEMPLATE}')

        self.descr = (
            'AgoraGalaxy weak-scaling benchmark using the standard weak-scaling '
            f"workflow (input: {INPUT_TEMPLATE.name})"
        )
        self.maintainers = ['@quokka-devs']
        self.tags = {'weak-scaling', 'agora-galaxy', 'performance'}

        self.sourcesdir = REPO_ROOT
        self.valid_systems = ['*']
        self.valid_prog_environs = ['*']

        self.build_system = 'CMake'
        self.build_system.builddir = 'build'
        self.build_system.config_opts = [
            '-G', 'Ninja',
            '-DCMAKE_BUILD_TYPE=Release',
            '-DAMReX_SPACEDIM=3',
        ]
        self.build_system.max_concurrency = 64
        self.build_system.build_targets = [
            'src/problems/AgoraGalaxy/agora_galaxy'
        ]

        self.executable = str(
            Path('build') / 'src' / 'problems' / 'AgoraGalaxy' / 'agora_galaxy'
        )
        self.input_filename = 'simulation.in'
        self.executable_opts = [self.input_filename]

        self.time_limit = self.scale['time_limit']

        self.num_tasks_per_node = TASKS_PER_NODE
        self.num_tasks = TASKS_PER_NODE * self.scale['nodes']
        self.num_cpus_per_task = CPUS_PER_TASK
        self.num_gpus_per_node = GPUS_PER_NODE

        self.reference = {
            '*': {
                'runtime': (0, None, 0, 's'),
                'zone_updates': (0, None, 0, 'cell-updates/s'),
                'tinyprofiler_avg': (0, None, 0, 's'),
            }
        }

    @run_before('compile')
    def configure_gpu_backend(self):
        backend = os.environ.get('QUOKKA_GPU_BACKEND')
        if not backend:
            features = set(getattr(self.current_partition, 'features', ()))
            if 'cuda' in features:
                backend = 'CUDA'
            elif 'hip' in features:
                backend = 'HIP'

        if backend:
            self.build_system.config_opts.append(f'-DAMReX_GPU_BACKEND={backend}')

    @run_before('run')
    def write_input_file(self):
        template = INPUT_TEMPLATE.read_text()
        cells = self.scale['cells_per_dim']

        template = re.sub(
            r'amr\.n_cell\s*=\s*\d+\s+\d+\s+\d+',
            f'amr.n_cell = {cells} {cells} {cells}',
            template,
        )
        template = template.replace('{{node_count}}', str(self.scale['nodes']))
        template = template.replace(
            '{{PREV_MAX_TIMESTEPS+100}}',
            str(self.scale['max_timesteps']),
        )

        Path(self.input_filename).write_text(template)

    @run_before('sanity')
    def set_sanity_patterns(self):
        self.sanity_patterns = sn.assert_found(
            r'TinyProfiler total time across processes', self.stdout
        )

    @run_before('performance')
    def set_perf_patterns(self):
        self.perf_patterns = {
            'runtime': self.wallclock_time,
            'zone_updates': self.zone_updates,
            'tinyprofiler_avg': sn.extractsingle(
                r'TinyProfiler total time across processes \[min\.\.\.avg\.\.\.max\]:\s+\S+\s+\.\.\.\s+(?P<avg>\S+)\s+\.\.\.\s+\S+',
                self.stdout,
                'avg',
                float,
            ),
        }

    @performance_function('s')
    def wallclock_time(self):
        return self.job.elapsed_time

    @performance_function('cell-updates/s')
    def zone_updates(self):
        cells = self.scale['cells_per_dim'] ** 3
        timesteps = self.scale['max_timesteps']
        return cells * timesteps / self.wallclock_time()

    @run_after('performance')
    def log_summary(self):
        total_cells = self.scale['cells_per_dim'] ** 3
        self.logger.info(
            'AgoraGalaxy weak-scaling (%s): %s nodes, %s MPI ranks, %s^3 cells (%s total), %s steps, '
            'runtime %.2f s, %.3e cell-updates/s',
            self.scale['label'],
            self.scale['nodes'],
            self.num_tasks,
            self.scale['cells_per_dim'],
            total_cells,
            self.scale['max_timesteps'],
            sn.evaluate(self.wallclock_time),
            sn.evaluate(self.zone_updates),
        )

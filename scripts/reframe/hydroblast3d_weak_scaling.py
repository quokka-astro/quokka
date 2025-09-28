import os

import reframe as rfm
import reframe.utility.sanity as sn
from reframe.core.builtins import parameter, performance_function, run_before, run_after


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

_BASE_CELLS_PER_DIM = 256
_BASE_NODES = 1
_TASKS_PER_NODE = 4
_GPUS_PER_NODE = 4
_SCALES_CONFIG = [
    ('n1_256', 256, '0:05:00'),
    ('n8_512', 512, '0:05:00'),
    ('n64_1024', 1024, '0:05:00'),
    ('n512_2048', 2048, '0:05:00'),
]

_WEAK_SCALES = []
for label, cells_per_dim, time_limit in _SCALES_CONFIG:
    scale_factor = cells_per_dim // _BASE_CELLS_PER_DIM
    nodes = _BASE_NODES * scale_factor ** 3
    _WEAK_SCALES.append(
        {
            'label': label,
            'nodes': nodes,
            'tasks_per_node': _TASKS_PER_NODE,
            'gpus_per_node': _GPUS_PER_NODE,
            'cells_per_dim': cells_per_dim,
            'input_file': os.path.join('inputs', f'benchmark_unigrid_{cells_per_dim}.in'),
            'time_limit': time_limit,
        }
    )


@rfm.simple_test
class HydroBlast3DWeakScalingTest(rfm.RegressionTest):
    scale = parameter(_WEAK_SCALES, fmt=lambda cfg: cfg['label'])

    def __init__(self):
        self.descr = (
            'HydroBlast3D weak-scaling benchmark with '
            f"{os.path.basename(self.scale['input_file'])}"
        )
        self.maintainers = ['@quokka-devs']
        self.tags = {'hydro', 'weak-scaling', 'performance'}
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
            'src/problems/HydroBlast3D/test_hydro3d_blast'
        ]

        self.executable = os.path.join(
            'build', 'src', 'problems', 'HydroBlast3D', 'test_hydro3d_blast'
        )
        self.executable_opts = [self.scale['input_file']]
        self.time_limit = self.scale['time_limit']

        self.num_tasks_per_node = self.scale['tasks_per_node']
        self.num_tasks = self.scale['tasks_per_node'] * self.scale['nodes']
        self.num_cpus_per_task = 1
        self.num_gpus_per_node = self.scale.get('gpus_per_node', 0)

        self.reference = {
            '*': {
                'runtime': (0, None, 0, 's'),
                'zone_updates': (0, None, 0, 'cell-updates/s'),
            }
        }

    @run_before('compile')
    def set_gpu_backend(self):
        backend = os.environ.get('QUOKKA_GPU_BACKEND')
        if not backend:
            features = set(getattr(self.current_partition, 'features', ()))
            if 'cuda' in features:
                backend = 'CUDA'
            elif 'hip' in features:
                backend = 'HIP'

        if backend:
            self.build_system.config_opts.append(f'-DAMReX_GPU_BACKEND={backend}')

    @run_before('sanity')
    def set_sanity_patterns(self):
        self.sanity_patterns = sn.assert_found(r'Writing plotfile', self.stdout)

    @run_before('performance')
    def set_perf_patterns(self):
        self.perf_patterns = {
            'runtime': self.wallclock_time,
            'zone_updates': self.zone_updates,
        }

    @performance_function('s')
    def wallclock_time(self):
        return self.job.elapsed_time

    @performance_function('cell-updates/s')
    def zone_updates(self):
        cells = self.scale['cells_per_dim'] ** 3
        return cells / self.wallclock_time()

    @run_after('performance')
    def log_summary(self):
        cells = self.scale['cells_per_dim'] ** 3
        self.logger.info(
            'HydroBlast3D weak-scaling (%s): %s cells, %s MPI ranks, runtime %.2f s, %.3e cell-updates/s',
            self.scale['label'],
            cells,
            self.num_tasks,
            sn.evaluate(self.wallclock_time),
            sn.evaluate(self.zone_updates),
        )

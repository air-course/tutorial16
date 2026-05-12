import os
import concurrent.futures
import multiprocessing

def _worker_fn(args):
    """Module-level worker for ProcessPoolExecutor (must be picklable)."""
    candidate, traj_opt_class, kwargs = args
    pid = os.getpid()
    traj_opt = traj_opt_class(
        xacro_out=f"/tmp/tmpArm_{pid}.xml",
        urdf_out=f"/tmp/parametricArm_{pid}.urdf",
        **kwargs,
    )
    fitness_val = traj_opt.fitness(candidate)
    return {
        'fitness': fitness_val,
        'params': traj_opt.logger['params'][-1],
        'xs': traj_opt.logger['xs'][-1],
        'qs': traj_opt.logger['qs'][-1],
        'us': traj_opt.logger['us'][-1],
        'gripperTrajectory': traj_opt.logger['gripperTrajectory'][-1],
        'gripperVelocity': traj_opt.logger['gripperVelocity'][-1],
    }


class ParallelObjective:
    """Callable returned by ParallelEvaluator.

    Pass as parallel_objective to cma.fmin2. Results accumulate in .logger.
    """

    def __init__(self, traj_opt_class, n_workers, **kwargs):
        self.traj_opt_class = traj_opt_class
        self.n_workers = n_workers
        self.kwargs = kwargs
        self.logger = {
            "params": [], "costs": [], "xs": [], "qs": [],
            "us": [], "gripperTrajectory": [], "gripperVelocity": []
        }

    def __call__(self, candidates):
        """Evaluate a whole CMA-ES population. Returns list of fitness values."""
        args = [(c, self.traj_opt_class, self.kwargs) for c in candidates]
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            results = list(executor.map(_worker_fn, args))

        fitnesses = []
        for r in results:
            fitnesses.append(r['fitness'])
            self.logger['params'].append(r['params'])
            self.logger['costs'].append(r['fitness'])
            self.logger['xs'].append(r['xs'])
            self.logger['qs'].append(r['qs'])
            self.logger['us'].append(r['us'])
            self.logger['gripperTrajectory'].append(r['gripperTrajectory'])
            self.logger['gripperVelocity'].append(r['gripperVelocity'])

        return fitnesses

    def scalar(self, candidate):
        """Single-candidate wrapper for the objective_function argument of cma.fmin2."""
        return self([candidate])[0]


class ParallelEvaluator:
    """Creates a parallel CMA-ES objective from any trajectory optimization class.

    Args:
        traj_opt_class: class to instantiate per candidate (e.g. TrajectoryOptimization)
        n_workers: number of parallel subprocesses (default: all CPU cores)

    Usage:
        evaluator = ParallelEvaluator(TrajectoryOptimization, n_workers=4)
        parallel_fn = evaluator(desired_position=desired_position)
        best_params, _ = cma.fmin2(parallel_fn.scalar, x0, sigma0, options,
                                   parallel_objective=parallel_fn)
        # results are in parallel_fn.logger
    """

    def __init__(self, traj_opt_class, n_workers=None):
        self.traj_opt_class = traj_opt_class
        self.n_workers = n_workers or multiprocessing.cpu_count()

    def __call__(self, **kwargs):
        """Return a ParallelObjective; kwargs are forwarded to the traj_opt_class constructor."""
        return ParallelObjective(self.traj_opt_class, self.n_workers, **kwargs)
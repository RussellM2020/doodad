import rllab.misc.logger as logger
from rllab.misc.ext import  set_seed
import os

def setup(seed , n_parallel, log_dir ):

    if seed is not None:
        set_seed(seed)

    if n_parallel > 0:
        from rllab.sampler import parallel_sampler
        parallel_sampler.initialize(n_parallel=n_parallel)
        if seed is not None:
            parallel_sampler.set_seed(seed)
    
    if os.path.isdir(log_dir)==False:
        os.makedirs(log_dir , exist_ok = True)

    logger.set_snapshot_dir(log_dir)
    #logger.set_snapshot_gap(20)

    logger.add_tabular_output(log_dir+'progress.csv')

    # params_log_file = osp.join(log_dir+'params.json')
    # logger.log_parameters_lite(params_log_file, args)
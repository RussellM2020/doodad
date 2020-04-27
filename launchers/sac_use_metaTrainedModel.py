import os

import doodad as dd
import doodad.ec2 as ec2
import doodad.ssh as ssh
import doodad.mount as mount
from doodad.utils import EXAMPLES_DIR, REPO_DIR

from launcher_utils import gen_mode, create_mounts , get_exp_name
import pickle


# envType = 'SawyerMultiDomain-pushDoorDrawer'  ; annotation = 'Push_aScale1_2' ;  max_path_length = 100
# n_train_tasks = 20; n_test_tasks = 10
# trainFile = 'multi_domain/multiDom_debugOnlyPush_aScale1_2'
# testFile  = 'multi_domain/multiDom_debugOnlyPush_aScale1_2_val'
# info_metric = 'placeDist'

envType = 'simple_pointMass' ; annotation = 'mean100-width10-aScale1-2' ; max_path_length = 20
n_train_tasks = 100 ; n_test_tasks = 20 ; info_metric = 'targetDist'
trainFile = 'pointMass/point_circle_mean100_width10_aScale1_2_v1'
testFile =  'pointMass/point_circle_mean100_width10_aScale1_2_v2'


# VERY IMPORTANT that n_train_tasks is >= the number of tasks in the trainFile, or we'll run into problems!
# TODO : change this to be safer
use_gpu = True
image = 'russellm888/smrl-gpu:latest'
log_dict = {'log_prefix':'sac_metaTrainedModel' , 'log_name':envType}
mode = 'docker'

run_mode = gen_mode(image, use_gpu , mode = mode , ssh_host = 'newton2' , s3 = log_dict )

output_dir =  '/home/code/mlc_project/smrl/smrl/data/'
mounts = create_mounts(code_dirs = ['doodad' , 'mlc_project/smrl' , 'rlkit', 'multiworld' , 'maml_rl' ] ,
                        output_dir = output_dir , ec2 = mode == 'ec2' )

THIS_FILE_DIR = os.path.realpath(os.path.dirname(__file__))
TRAIN = True # mode 0 for train , 1 for test
device = 'cuda' if use_gpu else 'cpu'

sampling_freq = 500
load_path = '/home/code/mlc_project/smrl/smrl/trained_models/pointMass_cDim2_numSteps1000_reg1_seed0_itr120/'
context_dim = 2
num_training_steps_per_epoch = 500 ; fast_adapt_steps = 5 ; task_id = 5

load_replay_buffers = True

dynamics_Weight = 1.0
for seed in [1,2]:

    expName = get_exp_name(
        annotation+'/Task_'+str(task_id)+'/samplingFreq_'+str(sampling_freq)+'_num_sacSteps_'+str(num_training_steps_per_epoch)+\
            '_fastAdaSteps_'+str(fast_adapt_steps)+'/seed_'+str(seed), log_dict = log_dict , mode = mode)
    dd.launch_python(
        #target=os.path.join(THIS_FILE_DIR, 'smrl.rlkit_main_remote'),
        #target = 'smrl.true_context_remote',
        target = 'smrl.main',
        python_cmd = 'python -m',

        #target=os.path.join(THIS_FILE_DIR, str(targetScript) ),  # point to a target script. If running remotely, this will be copied over
        mode=run_mode,
        mount_points=mounts,
        args={
            'variant': { 'TRAIN' : TRAIN, 'device' : device, 'seed': seed , 'remote': True, 'context_dim': context_dim,
                    'fast_adapt_steps' : fast_adapt_steps, 'info_metric': info_metric,

                    'train_sac_expert_use_metaTrainedModel' : True,
                    'expert_task_id' : task_id,
                    'envType': envType , 'max_path_length': max_path_length,  'train_task_file' : trainFile ,
                    'log_dir': output_dir+expName+'/' ,  'num_train_tasks': n_train_tasks ,

                    'sampling_freq': sampling_freq,
                    'num_training_steps_per_epoch' : num_training_steps_per_epoch, 'num_epochs' : 1000,
                    'fast_adapt_dynamics_loss_weight': dynamics_Weight,

                    'load_context_model': True ,  'load_sac_model' : True ,    'load_path':load_path  ,
                    'load_replay_buffers': load_replay_buffers
                } ,

            'output_dir': output_dir,
        }
    )



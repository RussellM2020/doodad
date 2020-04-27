import os

import doodad as dd
import doodad.ec2 as ec2
import doodad.ssh as ssh
import doodad.mount as mount
from doodad.utils import EXAMPLES_DIR, REPO_DIR

from launcher_utils import gen_mode, create_mounts , get_exp_name
import pickle

goal_dir= '/home/code/multiworld/multiworld/envs/goals/'

# envType = 'simple_pointMass' ; annotation = 'mean100-width10-aScale1-2' ; max_path_length = 20
# n_train_tasks = 100 ; n_test_tasks = 20 ; info_metric = 'targetDist'
# trainFile = 'pointMass/point_circle_mean100_width10_aScale1_2_v1'
# testFile =  'pointMass/point_circle_mean100_width10_aScale1_2_v2'
# dynamics_Weight = 1

envType = 'hc_vel' ; annotation = 'prior0_uniform_1' ; n_train_tasks =100 ; n_test_tasks = 20 ;  max_path_length = 200
trainFile = 'hc_vel/hc_vel_uniform_1' ; testFile  ='hc_vel/hc_vel_uniform_1'
info_metric = 'reward_forward' ; dynamics_Weight = 0

# envType = 'SawyerMultiDomain-pushDoorDrawer'  ; annotation = '25trajs_Push_aScale1_2' ;  max_path_length = 100
# n_train_tasks = 20; n_test_tasks = 10
# trainFile = 'multi_domain/multiDom_debugOnlyPush_aScale1_2'
# testFile  = 'multi_domain/multiDom_debugOnlyPush_aScale1_2_val'
# info_metric = 'placeDist'

use_gpu = True
image = 'russellm888/smrl-gpu:latest'
log_dict = {'log_prefix':'smrl_oldSac' , 'log_name':envType}
mode = 'docker'

run_mode = gen_mode(image, use_gpu , mode = mode , ssh_host = 'newton2' , s3 = log_dict )

output_dir =  '/home/code/mlc_project/smrl_oldSac/smrl/data/'
mounts = create_mounts(code_dirs = ['doodad' , 'mlc_project/smrl_oldSac' , 'rlkit', 'multiworld' , 'maml_rl'] ,
                        output_dir = output_dir , ec2 = mode == 'ec2' )

THIS_FILE_DIR = os.path.realpath(os.path.dirname(__file__))

TRAIN = True # mode 0 for train , 1 for test
device = 'cuda' if use_gpu else 'cpu'
collect_data_init_n_trajs = 10 ; prob_contexts = False ; debug_true_context = False
#If mode is test, need to load in replay buffers, context_model, sac_model
# if TRAIN:
load_path = '' ; load_context_model = False ; load_sac_model = False ; load_replay_buffers= False
#
# else:
#     load_path = exp_dir  + 'trained_models/'+load_dir
#     load_replay_buffers = True
#     load_context_model = True
#     load_sac_model = True
#dynamics_Weight = 1.0
num_training_steps_per_epoch = 1000
#num_sac_steps_per_model_step = 1
for seed in [0,1]:
    #for num_training_steps_per_epoch in [500, 1000]:
    for context_dim in [16]:
        #for context_regWeight in [ 0.1, 1]:
        #for num_sac_steps_per_model_step in [1, 5,10]:
        for context_regWeight in [0]:
    
            expName = get_exp_name(annotation +'/contextDim_'+str(context_dim)+'_steps_perEpoch_'+\
                             str(num_training_steps_per_epoch)+\
                            '_context_regWeight_'+str( context_regWeight )+\
                            '/seed_'+str(seed) ,  mode = mode, log_dict = log_dict)
            dd.launch_python(
                #target=os.path.join(THIS_FILE_DIR, 'smrl.rlkit_main_remote'),
                #target = 'smrl.true_context_remote',
                target = 'smrl.main',
                python_cmd = 'python -m',
                #target=os.path.join(THIS_FILE_DIR, str(targetScript) ),  # point to a target script. If running remotely, this will be copied over
                mode=run_mode,
                mount_points=mounts,
                args={
                    'variant': { 'TRAIN' : TRAIN, 'device' : device, 'seed': seed , 'remote' : True,

                            'envType': envType , 'max_path_length': max_path_length,  'train_task_file' : trainFile ,
                            'test_task_file' : testFile, 'log_dir':output_dir+expName+'/' ,
                            'num_train_tasks': n_train_tasks , 'num_test_tasks' : n_test_tasks, 'info_metric' : info_metric,

                            #'num_sac_steps_per_model_step' : num_sac_steps_per_model_step,
                            'num_training_steps_per_epoch' : num_training_steps_per_epoch ,  'collect_data_init_n_trajs' : collect_data_init_n_trajs,
                            'context_dim': context_dim , 'prob_contexts': prob_contexts , 'debug_true_context':debug_true_context,

                            'context_var_regWeight' : context_regWeight, 'fast_adapt_dynamics_loss_weight': dynamics_Weight,

                            'load_context_model' : load_context_model , 'load_sac_model': load_sac_model , 'load_path':load_path ,
                            'load_replay_buffers': load_replay_buffers
                        } ,
                    'output_dir': output_dir,
                }
            )



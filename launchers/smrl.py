import os

import doodad as dd
import doodad.ec2 as ec2
import doodad.ssh as ssh
import doodad.mount as mount
from doodad.utils import EXAMPLES_DIR, REPO_DIR

from launcher_utils import gen_mode, create_mounts , get_exp_name , list_to_str
import pickle

goal_dir= '/home/code/multiworld/multiworld/envs/goals/'

envType = 'simple_pointMass' ; annotation = 'mean100-width10-aScale1-2-prior0' ; max_path_length = 20
n_train_tasks = 100 ; n_test_tasks = 20 ; info_metric = 'targetDist'
trainFile = 'pointMass/point_circle_mean100_width10_aScale1_2_v1'
testFile =  'pointMass/point_circle_mean100_width10_aScale1_2_v2'

# envType = 'hc_vel' ; annotation = 'prior0_uniform_1' ; n_train_tasks =100 ; n_test_tasks = 20 ;  max_path_length = 200
# trainFile = 'hc_vel/hc_vel_uniform_1' ; testFile  ='hc_vel/hc_vel_uniform_1'
# info_metric = 'reward_forward'

# envType = 'cheetah-crippled' ; annotation = '' ; n_train_tasks = 5 ; n_test_tasks = 1 ;  max_path_length = 200
# info_metric = 'reward_run'

# envType = 'ant-crippled' ; annotation = 'fixedPrior0' ; n_train_tasks = 3 ; n_test_tasks = 1 ;  max_path_length = 200
# info_metric = 'reward_forward'


# envType = 'SawyerMultiDomain-pushDoorDrawer'  ; annotation = '25trajs_Push_aScale1_2' ;  max_path_length = 100
# n_train_tasks = 20; n_test_tasks = 10
# trainFile = 'multi_domain/multiDom_debugOnlyPush_aScale1_2'
# testFile  = 'multi_domain/multiDom_debugOnlyPush_aScale1_2_val'
# info_metric = 'placeDist'
use_gpu = True
image = 'russellm888/smrl-gpu:latest'
log_dict = {'log_prefix':'smrl_aug12' , 'log_name':envType}
mode = 'ssh'

run_mode = gen_mode(image, use_gpu , mode = mode , ssh_host = 'newton1' , s3 = log_dict )

output_dir =  '/home/code/mlc_project/smrl/smrl/data/'
mounts = create_mounts(code_dirs = ['doodad' , 'mlc_project/smrl' , 'rlkit', 'multiworld' ,
                                    'maml_rl' , 'learning_to_adapt'] ,
                        output_dir = output_dir , ec2 = mode == 'ec2' )

THIS_FILE_DIR = os.path.realpath(os.path.dirname(__file__))

TRAIN = True # mode 0 for train , 1 for test
device = 'cuda' if use_gpu else 'cpu'
collect_data_init_n_trajs = 10 ; prob_contexts = False ; debug_true_context = False
#If mode is test, need to load in replay buffers, context_model, sac_model
# if TRAIN:
load_path = '' ; load_context_model = False ; load_sac_model = False ; load_replay_buffers= False
# else:
#     load_path = exp_dir  + 'trained_models/'+load_dir
#     load_replay_buffers = True
#     load_context_model = True
#     load_sac_model = True

dynamics_Weight = 1.0
init_fast_learning_rate=1e-2
#fast_adapt_learning_rate=1e-2
#num_training_steps_per_epoch = 2000
#num_sac_steps_per_model_step = 1
num_training_steps_per_epoch = 1000
context_dim = 4
prior_posterior_trajs = (5, 45)
fast_adapt_steps = 2
context_regWeight = 10


for seed in [0,1]:
    #for fast_adapt_steps in [2,5]:
    for fast_adapt_learning_rate in [0.01, 0.03]:
            #for context_regWeight in [0,10]:

                # for fast_adapt_steps in [2,5]:
                #for context_regWeight in [ 0, 10]:
        expName = get_exp_name(annotation +'/contextDim_'+str(context_dim)+'_steps_perEpoch_'+str(num_training_steps_per_epoch)+\
                        #'/priorPosteriorTrajs_'+list_to_str(prior_posterior_trajs)+\
                        '/flr_'+str(fast_adapt_learning_rate)+\
                        '_fastAdaptSteps_'+str(fast_adapt_steps)+'_context_regWeight_'+str( context_regWeight )+\
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

                        'envType': envType , 'max_path_length': max_path_length, 'info_metric': info_metric,
                        'num_train_tasks': n_train_tasks, 'num_test_tasks': n_test_tasks,
                        'train_task_file' : trainFile , 'test_task_file' : testFile,
                        'log_dir':output_dir+expName+'/' ,

                        #'num_sac_steps_per_model_step' : num_sac_steps_per_model_step,
                        'num_training_steps_per_epoch' : num_training_steps_per_epoch ,
                        'collect_data_init_n_trajs' : collect_data_init_n_trajs,
                        #'collect_data_n_trajs_prior'    : prior_posterior_trajs[0],
                        #'collect_data_n_trajs_posterior': prior_posterior_trajs[1],

                        'context_dim': context_dim , 'prob_contexts': prob_contexts ,
                        'debug_true_context':debug_true_context,

                        'context_var_regWeight' : context_regWeight,
                        'dynamics_loss_weight': dynamics_Weight,
                        #'init_fast_learning_rate' : init_fast_learning_rate,
                        'fast_adapt_learning_rate' : fast_adapt_learning_rate,
                        'fast_adapt_steps' : fast_adapt_steps,

                        'load_context_model' : load_context_model , 'load_sac_model': load_sac_model ,
                        'load_path':load_path , 'load_replay_buffers': load_replay_buffers
                    } ,
                'output_dir': output_dir,
            }
        )
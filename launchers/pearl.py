import os

import doodad as dd
import doodad.ec2 as ec2
import doodad.ssh as ssh
import doodad.mount as mount
from doodad.utils import EXAMPLES_DIR, REPO_DIR

from launcher_utils import gen_mode, create_mounts , get_exp_name
import pickle


idx_metricMap = {}
###################### PEARL ###################################################

# envType = 'hc_vel' ; annotation = 'uniform_0.1' ; n_train_tasks =100 ; n_test_tasks = 20 ; max_path_length = 200
# trainFile = 'hc_vel/hc_vel_uniform_0.1'
# testFile = 'hc_vel/hc_vel_uniform_0.1'

# envType = 'cheetah-crippled' ; annotation = 'lastTest' ; n_train_tasks = 5 ; n_test_tasks = 1 ;  max_path_length = 200
# info_metric = 'reward_run'
# envType = 'ant-modControl' ; annotation = 'negJoints' ; n_train_tasks = 35 ; n_test_tasks = 35 ;  max_path_length = 200
# trainFile = 'ant/ant_negJoints_train_v2'  ; testFile = 'ant/ant_negJoints_test_v2'
# info_metric = ''

# envType = 'ant-qCircle' ; annotation = '' ; n_train_tasks = 100 ; n_test_tasks = 30 ;  max_path_length = 200
# trainFile = ''  ; testFile = ''
# info_metric = ''

# envType = 'humanoid-dir' ; annotation = 'restricted-dir-corrected' ; n_train_tasks = 10 ; n_test_tasks = 10 ;  max_path_length = 200
# trainFile = ''; testFile= ''
# info_metric = ''


envType = 'humanoid-modControl' ; annotation = 'dynamics-extrapolation' ; n_train_tasks = 10 ; n_test_tasks = 10 ;  max_path_length = 200
trainFile = 'humanoid/humanoid_dynamics_train'; testFile= 'humanoid/humanoid_dynamics_test'
info_metric = ''

# envType = 'cheetah-modControl' ; annotation = 'negJoints' ; n_train_tasks = 10 ; n_test_tasks = 10 ;  max_path_length = 200
# trainFile = 'hc/half_cheetah_neg_train_v2'  ; testFile = 'hc/half_cheetah_neg_test_v2'
# info_metric = ''


use_gpu = True
image = 'russellm888/smrl-gpu:latest'
log_dict = {'log_prefix':'pearl' , 'log_name':envType}
mode = 'ec2'

run_mode = gen_mode(image, use_gpu , mode = mode , ssh_host = 'newton2' , s3 = log_dict)

output_dir = '/home/code/oyster/output'
mounts = create_mounts(code_dirs = ['doodad' , 'rand_param_envs' , 'oyster', 'multiworld' , 'maml_rl' ,'MIER/smrl']
									,
						output_dir = output_dir , ec2 = mode == 'ec2' )


THIS_FILE_DIR = os.path.realpath(os.path.dirname(__file__))


train = True  ; uib = True
for seed in [0 , 1 ]:
	for ldim in [4 , 16]:
		for num_steps in [ 1000, 2000, 4000]:

			exp_name = get_exp_name(annotation+ '/ldim_'+str(ldim)+'_numSteps_'+str(num_steps)+\
									#'_posSteps_'+str(num_extra_rl_steps_posterior)+\
									'/seed_'+str(seed) ,
									mode = mode , log_dict = log_dict)

			dd.launch_python(
				target=os.path.join(THIS_FILE_DIR, '/home/russell/oyster/launch_experiment_custom.py'),
				mode = run_mode,
				mount_points = mounts,
				args={
					'variant':  {
									'remote' : True , 'train': True , 'seed' : seed , 'use_gpu': use_gpu,
									'envType' : envType , 'n_train_tasks' : n_train_tasks , 'n_test_tasks' : n_test_tasks,

									'trainFile' : trainFile ,  'testFile' : testFile,
									'latent_size' : ldim , 'exp_name' : exp_name,

									'use_trueContexts' : False , 'contextsFile' : None,
									#'path_to_weights': path_to_weights,
									#'sparse' : sparse,

									#'pushRew_weight': pushRew_weight,

								'algo_params':{
									# "num_steps_posterior": 0,
									'use_information_bottleneck': uib,
									'max_path_length': max_path_length ,
									'num_train_steps_per_itr' : num_steps,
									#'idx_metricMap': idx_metricMap,
									#'sparse_rewards': True,
									}
								},
					'output_dir': output_dir
				}
			)


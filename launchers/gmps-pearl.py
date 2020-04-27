import os

import doodad as dd
import doodad.ec2 as ec2
import doodad.ssh as ssh
import doodad.mount as mount
from doodad.utils import EXAMPLES_DIR, REPO_DIR
from launcher_utils import gen_mode, create_mounts , get_exp_name
import pickle
import numpy as np
###################### GMPS- PEARL ###################################################
traj_dir_prefix = '/home/code/gmps-oyster/saved_expert_trajs/'

envType = 'SawyerMultiDomain-pushDoorDrawer'  ; annotation = '10each-noPriorTaskKnowledge' ;  max_path_length = 100
n_train_tasks = 3; n_test_tasks = 0

trainFile = 'multi_domain/multiDomain_pushDoorDrawer_10each' ; contextsFile = None
expert_trajs_dir = traj_dir_prefix + 'SawyerMultiDomain-pushDoorDrawer-10each-numDemos20/Itr_400/'
family_priors = [[1,0] , [0,1] , [-1,0]]
family_metrics =  ['placeDist' , 'angleDelta' , 'posDelta']
########## Debug-door ##############################
# envType = 'SawyerMultiDomain-debug-Door' ; annotation = '10each' ; n_train_tasks =10 ; n_test_tasks = 1 ;
# max_path_length = 100
# trainFile = 'multi_domain/door_10each'
# contextsFile = None
# expert_trajs_dir = traj_dir_prefix + "SawyerMultiDomain-debug-Door/10_tasks/"

use_gpu = False
image = 'russellm888/smrl-gpu:latest'
log_dict = {'log_prefix':'gmps-pearl' , 'log_name':envType}
mode = 'docker'

run_mode = gen_mode(image, use_gpu , mode = mode , ssh_host = 'newton2' , s3 = log_dict )

output_dir = '/home/code/gmps-oyster/output'
mounts = create_mounts(code_dirs = ['doodad' , 'rand_param_envs' , 'gmps-oyster', 'multiworld' , 'maml_rl'] , 
						output_dir = output_dir , ec2 = mode == 'ec2' )


THIS_FILE_DIR = os.path.realpath(os.path.dirname(__file__))

train = True 
use_trueContexts = (contextsFile!=None)
ldim = 2
for meta_batch in [5, 10, 20]:
	for seed in [0 , 1]:
		for constrain_weight in [0.5, 1]:
			for num_steps in [1, 1000]:

				exp_name = get_exp_name(annotation+'/ldim_'+str(ldim)+'_numSteps_'+str(num_steps)+'_cweight_'+\
										str(constrain_weight)+ '_metaBatch_'+str(meta_batch)+'/seed_'+str(seed) ,
										mode = mode , log_dict = log_dict)

				dd.launch_python(
					target=os.path.join(THIS_FILE_DIR, '/home/russell/gmps-oyster/launch_experiment_custom.py'),
					mode = run_mode,
					mount_points = mounts,
					args={
						'variant':  {	'use_gpu': use_gpu,
										'remote' : True , 'train': train , 'seed' : seed ,
										 'use_trueContexts': use_trueContexts,

										'envType' : envType , 'n_train_tasks' : n_train_tasks ,
										'n_test_tasks' : n_test_tasks,
										'trainFile' : trainFile , 'contextsFile' : contextsFile,
										'family_priors' : family_priors,
										 'family_metrics': family_metrics,
										'latent_size' : ldim , 'exp_name' : exp_name,

									'algo_params':{
										'train_policy': True,
										'use_information_bottleneck': False,
										'constrain_weight': constrain_weight,
										'max_path_length': max_path_length ,
										'expert_trajs_dir': expert_trajs_dir,
										'num_train_steps_per_itr' : num_steps,
										'meta_batch': meta_batch
										}
									},
						'output_dir': output_dir
					}
				)

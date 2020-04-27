import os

import doodad as dd
import doodad.ec2 as ec2
import doodad.ssh as ssh
import doodad.mount as mount
from doodad.utils import EXAMPLES_DIR, REPO_DIR
import pickle


use_gpu = False
traj_dir_prefix = '/home/code/gmps-oyster/saved_expert_trajs/'

envType = 'SawyerMultiDomain-debug-Door' ; annotation = '10each' ; n_train_tasks =10 ; n_test_tasks = 1 ; max_path_length = 100
trainFile = 'multi_domain/door_10each'
contextsFile = None
expert_trajs_dir = traj_dir_prefix + "SawyerMultiDomain-debug-Door/10_tasks/" 

# envType = 'SawyerMultiDomain-push-door-drawer' ; annotation = '10each' ; n_train_tasks = 30 ; n_test_tasks = 1 ; max_path_length = 100
# trainFile = 'multi_domain/multiDomain_pushDoorDrawer_10each'
# contextsFile = 'push_door_drawer_priorDist_2'
# expert_trajs_dir = traj_dir_prefix + "SawyerMultiDomain-Push-Door-Drawer-pushDoorDrawer-10each-numDemos20/Itr_400/"



# envType = 'hc_vel' ; annotation = 'mean1-std1' ; n_train_tasks =20 ; n_test_tasks = 20 
# trainFile = 'hc_vel/hc_vel_mean1_std1_v1'

s3_log_prefix =  'gmps-pearl-gpu'
s3_log_name= envType+'-'+annotation 

#s3_log_name='smrl_fixed_num_tasks_true_context'
if use_gpu:
 
    mode_ec2 = dd.mode.EC2AutoconfigDocker(
            image='russellm888/smrl-gpu:latest',
            region='us-east-1',  # EC2 region
            instance_type='p2.xlarge',  # EC2 instance type
            spot_price=0.5,  # Maximum bid price
            s3_log_prefix = s3_log_prefix,
            s3_log_name=s3_log_name,
            gpu = True,
            terminate=True,  # Whether to terminate on finishing job
            extra_ec2_instance_kwargs = dict(
                    Placement=dict(
                        AvailabilityZone='us-east-1a',
                    ),
                )
        )
    mode_docker = dd.mode.LocalDocker(
    image = 'russellm888/smrl-gpu:latest',
    gpu = True
    )


else:
    mode_ec2 = dd.mode.EC2AutoconfigDocker(
            image= 'russellm888/mlc:latest' ,
            region='us-west-1',  # EC2 region
            instance_type= 'c4.2xlarge',  # EC2 instance type
            spot_price=0.4,  # Maximum bid price
            s3_log_prefix = s3_log_prefix,
            s3_log_name=s3_log_name,
            terminate=True,  # Whether to terminate on finishing job
        )

    mode_docker = dd.mode.LocalDocker(
    image = 'russellm888/mlc:latest',
    gpu = False
    )

#)

MY_RUN_MODE = mode_docker# CHANGE THIS

exp_dir = '/home/code/gmps-oyster/output/'
# Set up code and output directories
OUTPUT_DIR = exp_dir  # this is the directory visible to the target
#'/example/outputs' 
mounts = [

    mount.MountLocal(local_dir="~/doodad",
                         mount_point="/home/code/doodad",
                         filter_dir=["__pycache__", ".git"], pythonpath = True),


    mount.MountLocal(local_dir="~/rand_param_envs",
                         mount_point="/home/code/rand_param_envs/",
                         filter_dir=["__pycache__", ".git"], pythonpath = True),


    mount.MountLocal(local_dir="~/maml_rl",
                         mount_point="/home/code/maml_rl",
                         filter_dir=["__pycache__", ".git"], pythonpath = True),


    mount.MountLocal(local_dir="~/gmps-oyster",
                         mount_point="/home/code/gmps-oyster",
                         filter_dir=["__pycache__", ".git"], pythonpath = True),



    mount.MountLocal(local_dir="~/multiworld",
                         mount_point="/home/code/multiworld",
                         filter_dir=["__pycache__", ".git"], pythonpath = True),


    mount.MountLocal(local_dir="~/.mujoco",
                     mount_point="/root/.mujoco",
                     filter_dir=["__pycache__", ".git"]),
   
    ]

if MY_RUN_MODE == mode_ec2:
    output_mount = mount.MountS3(s3_path='', mount_point=OUTPUT_DIR, output=True)  # use this for ec2
else:
    output_mount = mount.MountLocal(local_dir=os.path.join(EXAMPLES_DIR, 'tmp_output'),
        mount_point=OUTPUT_DIR, output=True)
mounts.append(output_mount)

print(mounts)

THIS_FILE_DIR = os.path.realpath(os.path.dirname(__file__))


train = True 
use_trueContexts = (contextsFile!=None)
uib = True

for seed in [0]:
    for ldim in [4]:
    	for num_steps in [1]:

        	exp_name = 'ldim_'+str(ldim)+'_numSteps_'+str(num_steps)+'/seed_'+str(seed)
        #env_name = envType + '_train' if train else envType + '_test'
        
	        dd.launch_python(
	          
	            target=os.path.join(THIS_FILE_DIR, '/home/russell/gmps-oyster/launch_experiment_custom.py'),
	            mode=MY_RUN_MODE,
	            mount_points=mounts,
	            args={
	                'variant': {'remote' : True , 'train': train , 'seed' : seed , 'exp_name' : exp_name, 'use_trueContexts': use_trueContexts,
	                            'envType' : envType , 'trainFile' : trainFile , 'contextsFile' : contextsFile, 'latent_size' : ldim,
	                            'n_train_tasks' : n_train_tasks , 'n_test_tasks' : n_test_tasks,
	                            'algo_params' :{  "num_train_steps_per_itr" : num_steps,  
                                                'use_information_bottleneck': uib  , 'max_path_length': max_path_length , 'expert_trajs_dir': expert_trajs_dir} , 
	                           
	                'output_dir': OUTPUT_DIR,
	            }}
	        )
	      

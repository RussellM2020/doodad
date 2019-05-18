import os

import doodad as dd
import doodad.ec2 as ec2
import doodad.ssh as ssh
import doodad.mount as mount
from doodad.utils import EXAMPLES_DIR, REPO_DIR
import pickle


use_gpu = True

s3_log_prefix =  'smrl_benchmarking'
s3_log_name='no_relabelling' 

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

exp_dir = '/home/code/mlc_project/smrl_fixed_num_tasks/smrl/'
# Set up code and output directories
OUTPUT_DIR = exp_dir+'data/'   # this is the directory visible to the target
#'/example/outputs' 
mounts = [
#     mount.MountLocal(local_dir=REPO_DIR, pythonpath=True), # Code
#     mount.MountLocal(local_dir=os.path.join(EXAMPLES_DIR, 'secretlib'), pythonpath=True), # Code
# 
    mount.MountLocal(local_dir="~/doodad",
                         mount_point="/home/code/doodad",
                         filter_dir=["__pycache__", ".git"], pythonpath = True),


    mount.MountLocal(local_dir="~/mlc_project/smrl_fixed_num_tasks",
                         mount_point="/home/code/mlc_project/smrl_fixed_num_tasks/",
                         filter_dir=["__pycache__", ".git"], pythonpath = True),


    mount.MountLocal(local_dir="~/rlkit",
                         mount_point="/home/code/rlkit",
                         filter_dir=["__pycache__", ".git"], pythonpath = True),



    mount.MountLocal(local_dir="~/multiworld",
                         mount_point="/home/code/multiworld",
                         filter_dir=["__pycache__", ".git"], pythonpath = True),


    mount.MountLocal(local_dir="~/.mujoco",
                     mount_point="/home/.mujoco",
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

if use_gpu:
    device = 'cuda'
else:
    device = 'cpu'



max_path_length = 200 ; num_train_tasks = 2 ; num_test_tasks = 2; num_tasks = 2  ; num_training_steps_per_epoch = 2 ; collect_data_init_n_trajs = 10

env_type = 'hc_vel' ; num_train_tasks = 20 ; num_test_tasks = 20 ; num_tasks = 20 ; 
#Train indices : 0 -> num_train_tasks
#Test indices  : num_tasks - num_test_tasks -> num_tasks [same standard used in Pearl]


context_dim = 5 ; prob_contexts = False ; debug_true_context = False

mode = 1 # mode 0 for train , 1 for test
#If mode is test, need to load in replay buffers, context_model, sac_model
if mode == 0:
    load_path = '' ; load_context_model = False ; load_sac_model = False ; load_replay_buffers= False

elif mode == 1:
    load_path = exp_dir  + 'trained_models/hc_vel_100/'
    load_replay_buffers = True
    load_context_model = True
    load_sac_model = True

train_task_file = exp_dir +  'goals/hc_vel_mean1_std1_v1'
test_task_file = exp_dir +  'goals/hc_vel_mean1_std1_v1'



for seed in [0,1,2]:
    expName = env_type+'/loadDebug_clamped_preUpdate_diffBuffers/contextDim_'+str(context_dim)+'_steps_perEpoch_'+str(num_training_steps_per_epoch)+'/seed_'+str(seed)
    
    #load_path = OUTPUT_DIR+ expName+'/itr_20/'
    
    dd.launch_python(
        #target=os.path.join(THIS_FILE_DIR, 'smrl.rlkit_main_remote'),
        #target = 'smrl.true_context_remote',
        target = 'smrl.main_remote',
        python_cmd = 'python -m',

        #target=os.path.join(THIS_FILE_DIR, str(targetScript) ),  # point to a target script. If running remotely, this will be copied over
        mode=MY_RUN_MODE,
        mount_points=mounts,
        args={
            'variant': { 'mode' : mode, 'envType': env_type , 'context_dim': context_dim , 'prob_contexts': prob_contexts , 'log_dir':OUTPUT_DIR+expName+'/' ,
                        'train_task_file' : train_task_file , 'test_task_file' : test_task_file,

                        'collect_data_init_n_trajs' : collect_data_init_n_trajs, 'debug_true_context':debug_true_context,
                        'num_tasks': num_tasks,   'num_train_tasks': num_train_tasks , 'num_test_tasks' : num_test_tasks,
                        'seed': seed , 'num_training_steps_per_epoch' : num_training_steps_per_epoch , 'max_path_length': max_path_length, 
                        'device' : device,
                        'load_context_model' : load_context_model , 'load_sac_model': load_sac_model , 'load_path':load_path , 'load_replay_buffers': load_replay_buffers
                        } , 
                       
            'output_dir': OUTPUT_DIR,
        }
    )



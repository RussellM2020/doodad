import os

import doodad as dd
import doodad.ec2 as ec2
import doodad.ssh as ssh
import doodad.mount as mount
from doodad.utils import EXAMPLES_DIR, REPO_DIR
import pickle


use_gpu = False


mode_docker = dd.mode.LocalDocker(
    image='russellm888/mlc:latest',
    #image = 'russellm888/railrl-gpu:latest',
    gpu = use_gpu
)

s3_log_prefix =  'mlc_benchmarking'
s3_log_name='smrl_master' 

if use_gpu:
    s3_log_prefix +='_Images'
    mode_ec2 = dd.mode.EC2AutoconfigDocker(
            image='russellm888/mlc:latest',
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

#)

MY_RUN_MODE = mode_ec2# CHANGE THIS

# Set up code and output directories
OUTPUT_DIR = '/home/code/mlc_project/smrl_master/smrl/data/'   # this is the directory visible to the target
#'/example/outputs' 
mounts = [
#     mount.MountLocal(local_dir=REPO_DIR, pythonpath=True), # Code
#     mount.MountLocal(local_dir=os.path.join(EXAMPLES_DIR, 'secretlib'), pythonpath=True), # Code
# 
    mount.MountLocal(local_dir="~/doodad",
                         mount_point="/home/code/doodad",
                         filter_dir=["__pycache__", ".git"], pythonpath = True),


    mount.MountLocal(local_dir="~/mlc_project/smrl_master",
                         mount_point="/home/code/mlc_project/smrl_master/",
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

use_maesn = False



env_type = 'HalfCheetah_vel' ; context_dim = 5 ; prob_contexts = False ; num_tasks = 100
for collect_data_freq in [1000, 500]:
    for seed in [0,1,2]:
        expName = env_type+'/pearlParams_prob_'+str(prob_contexts)+'/collect_data_freq_'+str(collect_data_freq)+'/seed_'+str(seed)
        dd.launch_python(
            #target=os.path.join(THIS_FILE_DIR, 'smrl.rlkit_main_remote'),
            target = 'smrl.main_remote',
            python_cmd = 'python -m',

            #target=os.path.join(THIS_FILE_DIR, str(targetScript) ),  # point to a target script. If running remotely, this will be copied over
            mode=MY_RUN_MODE,
            mount_points=mounts,
            args={
                'variant': {'env_type': env_type , 'context_dim': context_dim , 'prob_contexts': prob_contexts , 'log_dir':OUTPUT_DIR+expName+'/' , 'num_tasks': num_tasks , 
                            'seed': seed , 'collect_data_freq': collect_data_freq},
                'output_dir': OUTPUT_DIR,
            }
        )



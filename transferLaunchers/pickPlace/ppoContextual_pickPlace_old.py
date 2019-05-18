import os

import doodad as dd
import doodad.ec2 as ec2
import doodad.ssh as ssh
import doodad.mount as mount
from doodad.utils import EXAMPLES_DIR, REPO_DIR
import pickle


# Local docker
mode_docker = dd.mode.LocalDocker(
    image='russellm888/ppo:v1',
)

# or this! Run experiment via docker on another machine through SSH


# or use this!
mode_ec2 = dd.mode.EC2AutoconfigDocker(
        image='russellm888/ppo:v1',
        region='us-west-1',  # EC2 region
        # instance_type='g2.2xlarge',  # EC2 instance type
        # spot_price=0.5,  # Maximum bid price
        instance_type= 'c4.2xlarge',  # EC2 instance type
        spot_price=0.15,  # Maximum bid price
        s3_log_prefix='Sawyer_PickPlace',  # Folder to store log files under
        s3_log_name='ppoContextual_20X20_6_8_diffNumTasks',
        terminate=True,  # Whether to terminate on finishing job
        
    )
#mode_ec2 = dd.mode.EC2AutoconfigDocker(
#    image='python:3.5',
#    region='us-west-1',
#    instance_type='m3.medium',
#    spot_price=0.02,
#)

MY_RUN_MODE = mode_ec2  # CHANGE THIS

# Set up code and output directories
OUTPUT_DIR = '/root/code/baselines/data/'   # this is the directory visible to the target
#'/example/outputs' 
mounts = [
#     mount.MountLocal(local_dir=REPO_DIR, pythonpath=True), # Code
#     mount.MountLocal(local_dir=os.path.join(EXAMPLES_DIR, 'secretlib'), pythonpath=True), # Code
# 
    

    mount.MountLocal(local_dir="~/doodad",
                         mount_point="/root/code/doodad",
                         filter_dir=["__pycache__", ".git"], pythonpath = True),


    mount.MountLocal(local_dir="~/baselines",
                         mount_point="/root/code/baselines",
                         filter_dir=["__pycache__", ".git"], pythonpath = True),

    mount.MountLocal(local_dir="~/multiworld",
                         mount_point="/root/code/multiworld",
                         filter_dir=["__pycache__", ".git"], pythonpath = True),

    mount.MountLocal(local_dir="~/.mujoco",
                     mount_point="/root/.mujoco",
                     filter_dir=["__pycache__", ".git"]),

    # mount.MountS3(s3_path="experiments",
    #               mount_point="/data/soroush/experiments",
    #               output=True),
    ]

if MY_RUN_MODE == mode_ec2:
    output_mount = mount.MountS3(s3_path='', mount_point=OUTPUT_DIR, output=True)  # use this for ec2
else:
    output_mount = mount.MountLocal(local_dir=os.path.join(EXAMPLES_DIR, 'tmp_output'),
        mount_point=OUTPUT_DIR, output=True)
mounts.append(output_mount)

print(mounts)

THIS_FILE_DIR = os.path.realpath(os.path.dirname(__file__))


for seed in range(5):
    for hidden_sizes in [(64, 64), (100, 100), (150, 100, 50)]:
        for batch_size in [2048,  20000]:
            for clipRange in [0.2, 0.05]:

        

                h_str = ''
                for i in hidden_sizes:
                    h_str+=str(i)+'_'

                expName = 'hiddenSizes_'+h_str+'bs_'+str(batch_size)+'clipRange_'+str(clipRange)+'seed_'+str(seed) 



                dd.launch_python(
                    target=os.path.join(THIS_FILE_DIR, '/home/russellm/baselines/baselines/ppo2/ppo_sawyer_pick_and_place.py'),
                    #target=os.path.join(THIS_FILE_DIR, str(targetScript) ),  # point to a target script. If running remotely, this will be copied over
                    mode=MY_RUN_MODE,
                    mount_points=mounts,
                    args={
                        'variant': {'seed': seed,  'batch_size': batch_size , 'contextual' : contextual , 'enableRotation' : enableRotation, 
                                    'hiddenSizes' : hidden_sizes, 'startTask': startTask, 'numTasks':1, 'expName' : expName},
                        
                        'output_dir': OUTPUT_DIR,
                    }
                )



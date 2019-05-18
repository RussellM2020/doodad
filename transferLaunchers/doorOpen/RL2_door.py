import os

import doodad as dd
import doodad.ec2 as ec2
import doodad.ssh as ssh
import doodad.mount as mount
from doodad.utils import EXAMPLES_DIR, REPO_DIR
import pickle

#russellm888/railrl-tf2
# Local docker
mode_docker = dd.mode.LocalDocker(
    image='russellm888/railrl-tf2:quat',
)

# or this! Run experiment via docker on another machine through SSH


# or use this!
mode_ec2 = dd.mode.EC2AutoconfigDocker(
        image='russellm888/railrl-tf2:quat',
        region='us-west-1',  # EC2 region
        # instance_type='g2.2xlarge',  # EC2 instance type
        # spot_price=0.5,  # Maximum bid price
        instance_type= 'c4.2xlarge',  # EC2 instance type
        spot_price=0.14,  # Maximum bid price
        # s3_log_prefix='Sawyer_pickPlace_finnMAML_20X20_6_8_normalized_diffLogging',  # Folder to store log files under
       
        s3_log_prefix='Sawyer_Door',  # Folder to store log files under
        s3_log_name='rl2_60X20X20',
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
OUTPUT_DIR = '/root/code/rllab-private-RL2unstable/data/'   # this is the directory visible to the target
#'/example/outputs' 
mounts = [
#     mount.MountLocal(local_dir=REPO_DIR, pythonpath=True), # Code
#     mount.MountLocal(local_dir=os.path.join(EXAMPLES_DIR, 'secretlib'), pythonpath=True), # Code
# 

    mount.MountLocal(local_dir="~/doodad",
                         mount_point="/root/code/doodad",
                         filter_dir=["__pycache__", ".git"], pythonpath = True),



    mount.MountLocal(local_dir="~/rllab-private-RL2unstable",
                         mount_point="/root/code/rllab-private-RL2unstable",
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





for seed in range(2):
    for clip_lr in [0.01, 0.05]:
        for numEpisodes in [2,5]:
            for batchSize in [50000, 150000]:
           
                expName = 'batchSize_'+str(batchSize)+'_numEp_'+str(numEpisodes)+'_seed'+str(seed)


                dd.launch_python(
                    target=os.path.join(THIS_FILE_DIR, '/home/russellm/rllab-private-RL2unstable/examples/rl2_sawyer_door.py'),
                    #target=os.path.join(THIS_FILE_DIR, str(targetScript) ),  # point to a target script. If running remotely, this will be copied over
                    mode=MY_RUN_MODE,
                    mount_points=mounts,
                    args={
                        'variant': {'saveDir':OUTPUT_DIR+expName+'/', 'seed':seed, 'n_episodes': numEpisodes, 'batch_size': batchSize,  'clip_lr' : clip_lr},
                        
                        'output_dir': OUTPUT_DIR,
                    }
                )



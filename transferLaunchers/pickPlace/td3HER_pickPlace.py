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
    image='russellm888/td3-her:quat',
)

# or this! Run experiment via docker on another machine through SSH


# or use this!
mode_ec2 = dd.mode.EC2AutoconfigDocker(
        image='russellm888/td3-her:quat',
        region='us-west-1',  # EC2 region
        # instance_type='g2.2xlarge',  # EC2 instance type
        # spot_price=0.5,  # Maximum bid price
        instance_type= 'c4.2xlarge',  # EC2 instance type
        spot_price=0.15,  # Maximum bid price
        s3_log_prefix='Sawyer_pickPlace_td3HER_20X20_6_8',  # Folder to store log files under
        
        terminate=True,  # Whether to terminate on finishing job
        
    )
#mode_ec2 = dd.mode.EC2AutoconfigDocker(
#    image='python:3.5',
#    region='us-west-1',
#    instance_type='m3.medium',
#    spot_price=0.02,
#)

MY_RUN_MODE = mode_docker  # CHANGE THIS

# Set up code and output directories
OUTPUT_DIR = '/root/code/railrl-private/data/'   # this is the directory visible to the target
#'/example/outputs' 
mounts = [
#     mount.MountLocal(local_dir=REPO_DIR, pythonpath=True), # Code
#     mount.MountLocal(local_dir=os.path.join(EXAMPLES_DIR, 'secretlib'), pythonpath=True), # Code
# 

    mount.MountLocal(local_dir="~/doodad",
                         mount_point="/root/code/doodad",
                         filter_dir=["__pycache__", ".git"], pythonpath = True),



    mount.MountLocal(local_dir="~/railrl-private",
                         mount_point="/root/code/railrl-private",
                         filter_dir=["__pycache__", ".git"], pythonpath = True),


    mount.MountLocal(local_dir="~/rllab",
                         mount_point="/root/code/rllab",
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


tasksFile = '/home/russellm/multiworld/multiworld/envs/goals/pickPlace_20X20_6_8.pkl'
tasks = pickle.load(open(tasksFile, 'rb'))


if MY_RUN_MODE == mode_ec2:

  
    mode_ec2.s3_log_name = 'ec2_td3HER'

                  
                


dd.launch_python(
    target=os.path.join(THIS_FILE_DIR, '/home/russellm/railrl-private/experiments/russell/td3HER_pickPlace.py'),
    #target=os.path.join(THIS_FILE_DIR, str(targetScript) ),  # point to a target script. If running remotely, this will be copied over
    mode=MY_RUN_MODE,
    mount_points=mounts,
    args={
        'variant': {'tasks': tasks},
        
        'output_dir': OUTPUT_DIR,
    }
)



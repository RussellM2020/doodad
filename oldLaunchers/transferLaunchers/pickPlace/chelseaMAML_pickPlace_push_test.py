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
#envType = 'Push'
envType = 'Push'
regionSize = '20X20'
#expType = ''
expType = '_Test_trainSet'
# or use this!
mode_ec2 = dd.mode.EC2AutoconfigDocker(
        image='russellm888/railrl-tf2:quat',
        region='us-west-2',  # EC2 region
        # instance_type='g2.2xlarge',  # EC2 instance type
        # spot_price=0.5,  # Maximum bid price
        instance_type= 'c4.2xlarge',  # EC2 instance type
        spot_price=0.3,  # Maximum bid price
        s3_log_prefix = 'Sawyer_'+envType+'_3D_block_v2',
        s3_log_name='noGoalInfo_finnMAML_'+regionSize+expType,
        terminate=True,  # Whether to terminate on finishing job
        
    )
#mode_ec2 = dd.mode.EC2AutoconfigDocker(
#    image='python:3.5',
#    region='us-west-1',
#    instance_type='m3.medium',
#    spot_price=0.02,
#)

MY_RUN_MODE = mode_ec2 # CHANGE THIS

# Set up code and output directories
OUTPUT_DIR = '/root/code/maml_rl/data/'   # this is the directory visible to the target
#'/example/outputs' 
mounts = [
#     mount.MountLocal(local_dir=REPO_DIR, pythonpath=True), # Code
#     mount.MountLocal(local_dir=os.path.join(EXAMPLES_DIR, 'secretlib'), pythonpath=True), # Code
# 

    mount.MountLocal(local_dir="~/doodad",
                         mount_point="/root/code/doodad",
                         filter_dir=["__pycache__", ".git"], pythonpath = True),



    mount.MountLocal(local_dir="~/maml_rl",
                         mount_point="/root/code/maml_rl",
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

#tasks = pickle.load(open('/home/russellm/multiworld/multiworld/envs/goals/PickPlace_20X20_simple.pkl', 'rb'))
if envType == 'Push':
    initFile = '/root/code/maml_rl/metaTrainedPolicies/push_20X20_v2.pkl'
elif envType == 'PickPlace':
    initFile = '/root/code/maml_rl/metaTrainedPolicies/pickPlace_20X20_v2.pkl'

init_step_size = 0.1 ; seed = 0

numTasks = 20
for index in range(numTasks):
#index = 5
    expName = 'seed_'+str(seed)+'/Task_'+str(index)
    dd.launch_python(
        target=os.path.join(THIS_FILE_DIR, '/home/russellm/maml_rl/maml_examples/pickPlace_push_test.py'),
        #target=os.path.join(THIS_FILE_DIR, str(targetScript) ),  # point to a target script. If running remotely, this will be copied over
        mode=MY_RUN_MODE,
        mount_points=mounts,
        args={
            'variant': {'goalIndex':index, 'initial_params_file': initFile, 'init_step_size' : init_step_size, 'saveDir':OUTPUT_DIR+expName+'/', 'seed' : seed,
                        'envType' : envType , 'regionSize' : regionSize },
            
            'output_dir': OUTPUT_DIR,
        }
    )



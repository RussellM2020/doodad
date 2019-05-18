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
    image='russellm888/railrl-tf2:mri',
)


envType = 'fixedDoor'
default_step = 0.1 ; policyType = 'biasAda'
# logSpecs = 'mbs5_'+policyType+ '_ldim2_defaultStep_'+str(default_step)
logSpecs= ''
mode_ec2 = dd.mode.EC2AutoconfigDocker(
        image='russellm888/railrl-tf2:mri',
        region='us-west-2',  # EC2 region
        # instance_type='g2.2xlarge',  # EC2 instance type
        # spot_price=0.5,  # Maximum bid price
        instance_type= 'c4.2xlarge',  # EC2 instance type
        spot_price=0.3,  # Maximum bid price
        s3_log_prefix = 'Sawyer_'+envType+'_MRI_offPol_Test',
        s3_log_name=logSpecs,
        terminate=True,  # Whether to terminate on finishing job
        
    )
#mode_ec2 = dd.mode.EC2AutoconfigDocker(
#    image='python:3.5',
#    region='us-west-1',
#    instance_type='m3.medium',
#    spot_price=0.02,
#)

MY_RUN_MODE = mode_docker # CHANGE THIS

# Set up code and output directories
OUTPUT_DIR = '/root/code/mri/data/'   # this is the directory visible to the target
#'/example/outputs' 
mounts = [
#     mount.MountLocal(local_dir=REPO_DIR, pythonpath=True), # Code
#     mount.MountLocal(local_dir=os.path.join(EXAMPLES_DIR, 'secretlib'), pythonpath=True), # Code
# 

    mount.MountLocal(local_dir="~/doodad",
                         mount_point="/root/code/doodad",
                         filter_dir=["__pycache__", ".git"], pythonpath = True),

    mount.MountLocal(local_dir="~/mri",
                         mount_point="/root/code/mri",
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

n_parallel = 1  ; n_itr = 2

#initFile = '/root/code/maml_rl/metaTrainedPolicies/Sawyer-fixedDoor/' + logSpecs + '.pkl'
initFile = '/root/code/mri/metaTrainedPolicies/itr_0.pkl'

# for val in [False , True]:
#     for seed in range(2):
#         for index in range(5):

val = False ; seed = 0 ; index = 3 ; 

if val :
    tasksFile = 'fixedDoor_diffAngles_val'
else:
    tasksFile = 'fixedDoor_diffAngles'

expName = 'Val_'+str(val)+'/Task_'+str(index)+'/Seed_'+str(seed)
dd.launch_python(
    target=os.path.join(THIS_FILE_DIR, '/home/russell/mri/maml_examples/remote_test.py'),
    #target=os.path.join(THIS_FILE_DIR, str(targetScript) ),  # point to a target script. If running remotely, this will be copied over
    mode=MY_RUN_MODE,
    mount_points=mounts,
    args={
        'variant': {'taskIndex':index, 'init_file': initFile,  'n_parallel' : n_parallel ,   'log_dir':OUTPUT_DIR+expName+'/', 'seed' : seed  , 'tasksFile' : tasksFile ,
                    'policyType' : policyType ,  'n_itr' : n_itr , 'default_step' : default_step , 'envType' : envType},
        
        'output_dir': OUTPUT_DIR,
    }
)



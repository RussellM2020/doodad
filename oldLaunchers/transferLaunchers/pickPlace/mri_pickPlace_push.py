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


envType = 'Push'
regionSize = '20X20'
# or this! Run experiment via docker on another machine through SSH


# or use this!
mode_ec2 = dd.mode.EC2AutoconfigDocker(
        image='russellm888/railrl-tf2:quat',
        region='us-west-2',  # EC2 region
        # instance_type='g2.2xlarge',  # EC2 instance type
        # spot_price=0.5,  # Maximum bid price
        instance_type= 'c4.2xlarge',  # EC2 instance type
        spot_price=0.5,  # Maximum bid price
        # s3_log_prefix='Sawyer_pickPlace_finnMAML_20X20_6_8_normalized_diffLogging',  # Folder to store log files under
       
        #s3_log_prefix='Sawyer_PickPlace_3D_block', 
        s3_log_prefix = 'Sawyer_'+envType+ '_3D_block',
        s3_log_name='mri_onPolicy_'+regionSize,
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
OUTPUT_DIR = '/root/code/mri_onPolicy/data/'   # this is the directory visible to the target
#'/example/outputs' 
mounts = [
#     mount.MountLocal(local_dir=REPO_DIR, pythonpath=True), # Code
#     mount.MountLocal(local_dir=os.path.join(EXAMPLES_DIR, 'secretlib'), pythonpath=True), # Code
# 

    mount.MountLocal(local_dir="~/doodad",
                         mount_point="/root/code/doodad",
                         filter_dir=["__pycache__", ".git"], pythonpath = True),



    mount.MountLocal(local_dir="~/mri_onPolicy",
                         mount_point="/root/code/mri_onPolicy",
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

hidden_sizes = (100, 100)
mlr = 0.01
seed = 0
if envType == 'Push':
    expertDataLoc = '/root/code/mri_onPolicy/expertPolicyWeights/TRPO-push-20X20/'
    #expertDataItr = 250

#for seed in range(3):
    
#    for fbs in [7 , 17, 50]:
#        for flr in [0.05,  0.2]:
n_parallel = 12

for expertDataItr in [10, 20, 30, 40]:
    for fbs in [20, 50]:
        for flr in [0.05, 0.1, 0.5]:
            
            expName = 'targetItr_'+str(expertDataItr)+'mlr_'+str(mlr)+'_flr_'+str(flr)+'_fbs_'+str(fbs)+'_seed'+str(seed)
            dd.launch_python(
                target=os.path.join(THIS_FILE_DIR, '/home/russellm/mri_onPolicy/maml_examples/maml_sawyer_pickPlace_push_launcher.py'),
                #target=os.path.join(THIS_FILE_DIR, str(targetScript) ),  # point to a target script. If running remotely, this will be copied over
                mode=MY_RUN_MODE,
                mount_points=mounts,
                args={
                    'variant': {'fbs':fbs, 'flr':flr, 'mlr': mlr, 'hidden_sizes':hidden_sizes, 'log_dir':OUTPUT_DIR+expName+'/', 'seed': seed , 'regionSize': regionSize, 
                    'envType': envType,  'n_parallel' : n_parallel , 'expertDataLoc': expertDataLoc, 'expertDataItr': expertDataItr},
                    
                    'output_dir': OUTPUT_DIR,
                }
            )



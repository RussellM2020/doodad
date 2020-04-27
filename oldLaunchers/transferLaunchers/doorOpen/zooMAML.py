import os

import doodad as dd
import doodad.ec2 as ec2
import doodad.ssh as ssh
import doodad.mount as mount
from doodad.utils import EXAMPLES_DIR, REPO_DIR
import pickle

#russelllsm888/railrl-tf2
# Local docker
mode_docker = dd.mode.LocalDocker(
    image='russellm888/maml_zoo:first',
)

# or this! Run experiment via docker on another machine through SSH

envType = 'Push'
regionSize = '20X20'
# or use this!

if envType == 'Push' or envType == 'PickPlace':

    prefixEnvName =  envType+'-3D-block'
else:
    prefixEnvName = envType

mode_ec2 = dd.mode.EC2AutoconfigDocker(
        image='russellm888/maml_zoo:first',
        region='us-west-1',  # EC2 region
        # instance_type='g2.2xlarge',  # EC2 instance type
        # spot_price=0.5,  # Maximum bid price
        instance_type= 'c4.8xlarge',  # EC2 instance type
        spot_price=0.5,  # Maximum bid price
        s3_log_prefix='Sawyer-'+prefixEnvName,  # Folder to store log files under

        s3_log_name='zooMAML_'+regionSize,
        
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
OUTPUT_DIR = '/root/code/maml_zoo/data/'   # this is the directory visible to the target
#'/example/outputs' 
mounts = [
#     mount.MountLocal(local_dir=REPO_DIR, pythonpath=True), # Code
#     mount.MountLocal(local_dir=os.path.join(EXAMPLES_DIR, 'secretlib'), pythonpath=True), # Code
# 

    mount.MountLocal(local_dir="~/doodad",
                         mount_point="/root/code/doodad",
                         filter_dir=["__pycache__", ".git"], pythonpath = True),



    mount.MountLocal(local_dir="~/maml_zoo",
                         mount_point="/root/code/maml_zoo",
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

#hidden_sizes = (100, 100)
#'init_inner_kl_penalty': 0.001,


learning_rate = 0.01 

for seed in range(2):
    
    for rollouts_per_meta_task in [ 20, 50]:
    #for mlr in 
        for inner_lr in [0.01, 0.05]:

        
           
            expName = 'fbs_'+str(rollouts_per_meta_task)+'_innerLR_'+str(inner_lr)+'_LR_'+str(learning_rate)+'/seed_'+str(seed)




            dd.launch_python(
                target=os.path.join(THIS_FILE_DIR, '/home/russellm/maml_zoo/run_scripts/ppoMaml.py'),
                #target=os.path.join(THIS_FILE_DIR, str(targetScript) ),  # point to a target script. If running remotely, this will be copied over
                mode=MY_RUN_MODE,
                mount_points=mounts,
                args={
                    'variant': {'rollouts_per_meta_task':rollouts_per_meta_task,  'data_path': OUTPUT_DIR+expName+'/', 'seed': seed, 
                                'inner_lr': inner_lr, 'learning_rate': learning_rate, 'envType' : envType, 'regionSize' : regionSize},
                    
                    'output_dir': OUTPUT_DIR,
                }
            )




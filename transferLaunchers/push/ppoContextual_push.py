import os

import doodad as dd
import doodad.ec2 as ec2
import doodad.ssh as ssh
import doodad.mount as mount
from doodad.utils import EXAMPLES_DIR, REPO_DIR
import pickle


# Local docker
mode_docker = dd.mode.LocalDocker(
    image='russellm888/ppo:quat',
)

# or this! Run experiment via docker on another machine through SSH


# or use this!
mode_ec2 = dd.mode.EC2AutoconfigDocker(
        image='russellm888/ppo:quat',
        region='us-east-1',  # EC2 region
        # instance_type='g2.2xlarge',  # EC2 instance type
        # spot_price=0.5,  # Maximum bid price
        instance_type= 'c4.2xlarge',  # EC2 instance type
        spot_price=0.15,  # Maximum bid price
        s3_log_prefix='Sawyer_Push_3D_block',  # Folder to store log files under
        s3_log_name='ppoContextual_60X30_block_size0.04_mass0.01_Rew_posPlace',
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



contextual = True ; enableRotation = False ; rewMode = 'posPlace' ; objType = 'block'
#seed = 0 ; hidden_sizes = [150, 100, 50] ; batch_size = 2048 ; startTask = 0
cliprange = 0.2 ; batch_size = 2048
startTask = 0; numTasks = 20

#for npSeed in range(5):
for npSeed in range(3):

    for hidden_sizes in [(100,100), (64,64)]:

    
    # for startTask in range(0 , 7, 14):
    #     for numTasks in range(2,5):

        tfSeed = npSeed
     
        h_str = ''
        for i in hidden_sizes:
            h_str+=str(i)+'_'

        expName = 'hiddenSizes_'+h_str+'bs_'+str(batch_size)+'_npSeed'+str(npSeed)+'_tfSeed'+str(tfSeed)+'_clipRange'+str(cliprange)



        dd.launch_python(
            target=os.path.join(THIS_FILE_DIR, '/home/russellm/baselines/baselines/ppo2/ppo_sawyer_push.py'),
            #target=os.path.join(THIS_FILE_DIR, str(targetScript) ),  # point to a target script. If running remotely, this will be copied over
            mode=MY_RUN_MODE,
            mount_points=mounts,
            args={
                'variant': {'tfSeed': tfSeed, 'npSeed' : npSeed,  'batch_size': batch_size , 'contextual' : contextual , 'enableRotation' : enableRotation, 
                    'rewMode': rewMode, 'objType' : objType, 'hiddenSizes' : hidden_sizes, 'startTask': startTask, 'numTasks':numTasks, 'expName' : expName, 'cliprange': cliprange},
                
                'output_dir': OUTPUT_DIR,
            }
        )



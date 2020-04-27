import os

import doodad as dd
import doodad.ec2 as ec2
import doodad.ssh as ssh
import doodad.mount as mount
from doodad.utils import EXAMPLES_DIR, REPO_DIR
import pickle

#russellm888/railrl-tf2
# Local docker


envType = 'Push' ; use_gpu = False
s3_log_prefix =  'Sawyer_'+envType
s3_log_name='mri_maesn'

mode_docker = dd.mode.LocalDocker(
    image='russellm888/railrl-gpu:tf',
    gpu = use_gpu
)

if use_gpu:
    mode_ec2 = dd.mode.EC2AutoconfigDocker(
            image='russellm888/railrl-gpu:tf',
            region='us-east-1',  # EC2 region
            instance_type='g3.16xlarge',  # EC2 instance type
            spot_price=1.0,  # Maximum bid price
         
            s3_log_prefix = s3_log_prefix+'_Images',
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
            image= 'russellm888/railrl-gpu:tf' ,
            region='us-west-1',  # EC2 region
            instance_type= 'c4.2xlarge',  # EC2 instance type
            spot_price=0.5,  # Maximum bid price
            s3_log_prefix = s3_log_prefix,
            s3_log_name=s3_log_name,
            terminate=True,  # Whether to terminate on finishing job
           
        )
#mode_ec2 = dd.mode.EC2AutoconfigDocker(
#    image='python:3.5',
#    region='us-west-1',
#    instance_type='m3.medium',
#    spot_price=0.02,
#)

MY_RUN_MODE = mode_docker# CHANGE THIS

# Set up code and output directories
OUTPUT_DIR = '/home/code/mri/data/'   # this is the directory visible to the target
#'/example/outputs' 
mounts = [
#     mount.MountLocal(local_dir=REPO_DIR, pythonpath=True), # Code
#     mount.MountLocal(local_dir=os.path.join(EXAMPLES_DIR, 'secretlib'), pythonpath=True), # Code
# 

    mount.MountLocal(local_dir="~/doodad",
                         mount_point="/home/code/doodad",
                         filter_dir=["__pycache__", ".git"], pythonpath = True),


    mount.MountLocal(local_dir="~/mri",
                         mount_point="/home/code/mri",
                         filter_dir=["__pycache__", ".git"], pythonpath = True),



    mount.MountLocal(local_dir="~/multiworld",
                         mount_point="/home/code/multiworld",
                         filter_dir=["__pycache__", ".git"], pythonpath = True),

    mount.MountLocal(local_dir="~/.mujoco",
                     mount_point="/home/.mujoco",
                     filter_dir=["__pycache__", ".git"]),

    # mount.MountS3(s3_path="data",
    #               mount_point="/home/data",
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


# expertDataItr = 250
# expertDataLoc = '/home/code/mri/saved_expert_trajs/imageObs-Sawyer-Pusher-3D-v1/Itr_250/'
# policyType = 'conv'

expertDataItr = 400
expertDataLoc = '/home/code/mri/saved_expert_trajs/Sawyer-Pusher-3D-v1/'
policyType = 'maesn'

init_flr = 0.1 ; ldim= 2 ; mbs = 20 ; tasksFile = 'pickPlace_20X20_v1' ; use_maesn = True
seed = 0 ; fbs = 20  ; adam_steps = 200


expName = 'mbs_'+str(mbs)+'/Maesn_policyType_'+policyType+'/adamSteps_'+str(adam_steps)+'_fbs_'+str(fbs)+'ldim_'+str(ldim)+'_initFlr_'+str(init_flr)+'_seed_'+str(seed)
dd.launch_python(
    target=os.path.join(THIS_FILE_DIR, '/home/russell/mri/maml_examples/remote_maml_il.py'),
    #target=os.path.join(THIS_FILE_DIR, str(targetScript) ),  # point to a target script. If running remotely, this will be copied over
    mode=MY_RUN_MODE,
    mount_points=mounts,
    args={
        'variant': {'policyType':policyType, 'ldim':ldim, 'init_flr': init_flr, 'seed' : seed , 'log_dir':OUTPUT_DIR+expName+'/',  
        'expertDataLoc': expertDataLoc ,   'envType': envType  , 'fbs' : fbs  , 'mbs' : mbs , 'tasksFile' : tasksFile , 'adam_steps' : adam_steps , 'use_maesn' : use_maesn},
        
        'output_dir': OUTPUT_DIR,
    }
)



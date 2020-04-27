import os

import doodad as dd
import doodad.ec2 as ec2
import doodad.ssh as ssh
import doodad.mount as mount
from doodad.utils import EXAMPLES_DIR, REPO_DIR
import pickle


use_gpu = True


mode_docker = dd.mode.LocalDocker(
    image='lauramsmith/umbrellas:cuda9',
    gpu = use_gpu
)

s3_log_prefix = 'umbrellas'
s3_log_name='trial-repeated'

if use_gpu:
    s3_log_prefix +='_Images'
    mode_ec2 = dd.mode.EC2AutoconfigDocker(
            image='lauramsmith/umbrellas:cuda9',
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
            image= 'lauramsmith/umbrellas:latest' ,
            region='us-west-2',  # EC2 region
            instance_type= 'c4.2xlarge',  # EC2 instance type
            spot_price=0.4,  # Maximum bid price
            s3_log_prefix = s3_log_prefix,
            s3_log_name=s3_log_name,
            terminate=True,  # Whether to terminate on finishing job
        )

#)

MY_RUN_MODE = mode_ec2# CHANGE THIS

# Set up code and output directories
OUTPUT_DIR = '/home/code/umbrellas/vae'   # this is the directory visible to the target
#'/example/outputs' 
mounts = [
#     mount.MountLocal(local_dir=REPO_DIR, pythonpath=True), # Code
#     mount.MountLocal(local_dir=os.path.join(EXAMPLES_DIR, 'secretlib'), pythonpath=True), # Code
# 
    mount.MountLocal(local_dir="~/doodad",
                         mount_point="/home/code/doodad",
                         filter_dir=["__pycache__", ".git"], pythonpath = True),


    mount.MountLocal(local_dir="~/laura/umbrellas2",
                         mount_point="/home/code/umbrellas",
                         filter_dir=["__pycache__", ".git"], pythonpath = True),



    mount.MountLocal(local_dir="~/laura/R_multiworld",
                         mount_point="/home/code/R_multiworld",
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

# if MY_RUN_MODE == mode_docker:
#     expName = s3_log_prefix+'/'+s3_log_name+'/'+expName

# '/home/lauramsmith/dev/umbrellas/scripts/debug_state_control.py debug_state 3 3 --num-traj 1 --model-iter 0 --mpc-policy \
#                                         --sparse --init-params vae/debug_state/weights/model-100000-weights.pkl --init-dynamics vae/debug_state/weights/model-300000-dynamics.pkl --horizon 30 --state'
#python 
dd.launch_python(
    target=os.path.join(THIS_FILE_DIR, '/home/russell/laura/umbrellas2/scripts/test_onemodel_remote.py exp_name a 12 15 --sparse --batch-size 1'

    ),
    
    mode=MY_RUN_MODE,
    mount_points=mounts,
    args={
        'variant': {},
        
        'output_dir': OUTPUT_DIR,
    }
)

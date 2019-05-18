import os

import doodad as dd
import doodad.ec2 as ec2
import doodad.ssh as ssh
import doodad.mount as mount
from doodad.utils import EXAMPLES_DIR, REPO_DIR
import pickle


# Local docker
mode_docker = dd.mode.LocalDocker(
    image='russellm888/railrl-tf2:quat',
)
regionSize = '20X20'

# or this! Run experiment via docker on another machine through SSH


# or use this!
mode_ec2 = dd.mode.EC2AutoconfigDocker(
        image='russellm888/railrl-tf2:quat',
        region='us-west-1',  # EC2 region
        # instance_type='g2.2xlarge',  # EC2 instance type
        # spot_price=0.5,  # Maximum bid price
        instance_type= 'c4.2xlarge',  # EC2 instance type
        spot_price=0.15,  # Maximum bid price
        s3_log_prefix='Sawyer_Push_3D_block',  # Folder to store log files under
        s3_log_name='Repeat-trpoExperts_'+regionSize+ '_block_size0.04_mass0.01',
        terminate=True,  # Whether to terminate on finishing job
        
    )

MY_RUN_MODE = mode_ec2 # CHANGE THIS

# Set up code and output directories
OUTPUT_DIR = '/root/code/maml_rl_baseline_test/data/'   # this is the directory visible to the target
#'/example/outputs' 
mounts = [
#     mount.MountLocal(local_dir=REPO_DIR, pythonpath=True), # Code
#     mount.MountLocal(local_dir=os.path.join(EXAMPLES_DIR, 'secretlib'), pythonpath=True), # Code
# 
    mount.MountLocal(local_dir="~/doodad",
                         mount_point="/root/code/doodad",
                         filter_dir=["__pycache__", ".git"], pythonpath = True),

    mount.MountLocal(local_dir="~/maml_rl_baseline_test",
                         mount_point="/root/code/maml_rl_baseline_test",
                         filter_dir=["__pycache__", ".git"], pythonpath = True),

    mount.MountLocal(local_dir="~/multiworld",
                         mount_point="/root/code/multiworld",
                         filter_dir=["__pycache__", ".git"], pythonpath = True),

    mount.MountLocal(local_dir="~/.mujoco",
                     mount_point="/root/.mujoco",
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



contextual = False ; enableRotation = False ; rewMode = 'posPlace'
#seed = 0 ; hidden_sizes = [150, 100, 50] ; batch_size = 2048 ; startTask = 0
seed = 1 ; n_parallel = 8 ; envClass = 'SawyerPusher' ; rate = 0.01
#args['variant'] = {'seed' : 1, 'n_parallel' : 8 , 'log_dir': '/home/russellm/data/TRPO-SawyerPusher-New', 'envClass' : 'SawyerPusher' , 'reset_arg' : 2, 'rate': 0.01}

for reset_arg in range(1):
    expName = 'Task_'+str(reset_arg)

    dd.launch_python(
        target=os.path.join(THIS_FILE_DIR, '/home/russellm/maml_rl_baseline_test/iclr18/trpo_launcher.py'),
        #target=os.path.join(THIS_FILE_DIR, str(targetScript) ),  # point to a target script. If running remotely, this will be copied over
        mode=MY_RUN_MODE,
        mount_points=mounts,
        args={

            'variant' : {'seed' : seed, 'n_parallel' : n_parallel , 'log_dir': OUTPUT_DIR+expName+'/', 'envClass' : 'SawyerPusher' , 'reset_arg' : reset_arg, 'rate': 0.01},           
            'output_dir': OUTPUT_DIR,
        }
    )

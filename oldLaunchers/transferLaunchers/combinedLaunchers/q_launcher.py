import os

import doodad as dd
import doodad.ec2 as ec2
import doodad.ssh as ssh
import doodad.mount as mount
from doodad.utils import EXAMPLES_DIR, REPO_DIR
import pickle

docker_image = 'russellm888/railrl-gpu:latest'
# Local docker
mode_docker = dd.mode.LocalDocker(
    image=docker_image,
)

#env_name = 'ndim-pointMass' ; ndim = 1

env_name = 'half-cheetah' ; ndim = 1

if 'pointMass' in env_name:
    s3_log_name = str(ndim)+'dim-pointMass'
else:
    s3_log_name = env_name

mode_ec2 = dd.mode.EC2AutoconfigDocker(
        image=docker_image,
        region='us-west-1',  # EC2 region
        # instance_type='g2.2xlarge',  # EC2 instance type
        # spot_price=0.5,  # Maximum bid price
        instance_type= 'c4.2xlarge',  # EC2 instance type
        spot_price=0.4,  # Maximum bid price
        s3_log_prefix='AMQ',  # Folder to store log files under
        s3_log_name=s3_log_name,
        terminate=True,  # Whether to terminate on finishing job
        
    )

MY_RUN_MODE = mode_docker # CHANGE THIS

# Set up code and output directories
OUTPUT_DIR = '/root/code/AMQ/data/'   # this is the directory visible to the target
#'/example/outputs' 

mounts = [
#     mount.MountLocal(local_dir=REPO_DIR, pythonpath=True), # Code
#     mount.MountLocal(local_dir=os.path.join(EXAMPLES_DIR, 'secretlib'), pythonpath=True), # Code
# 
    mount.MountLocal(local_dir="~/doodad",
                         mount_point="/root/code/doodad",
                         filter_dir=["__pycache__", ".git"], pythonpath = True),

    mount.MountLocal(local_dir="~/AMQ",
                         mount_point="/root/code/AMQ",
                         filter_dir=["__pycache__", ".git"], pythonpath = True),

    mount.MountLocal(local_dir= "~/rllab",
                         mount_point="/root/code/rllab",
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

#algo = 'naf-intercept-1q' 

#algo = 'ddpg-1q'
#for algo in ['naf-intercept-1q' , 'naf-std-1q' , 'amq-intercept-1q' , 'amq-std-1q' , 'ddpg-1q']:
algo = 'amq-intercept-1q'
for seed in [0,1]:
    for learning_rate in [3e-4 ]:
        for target_update_freq in [ 500]:

            expName = 'algo_'+algo+'/lr_'+str(learning_rate)+'_target_update_freq_'+str(target_update_freq)+'/seed_'+str(seed)
            dd.launch_python(
                target=os.path.join(THIS_FILE_DIR, '/home/russell/AMQ/examples/q_launcher.py'),
                #target=os.path.join(THIS_FILE_DIR, str(targetScript) ),  # point to a target script. If running remotely, this will be copied over
                mode=MY_RUN_MODE,
                mount_points=mounts,
                args={
                   
                    'variant' : {'log_dir' : OUTPUT_DIR+expName+'/', 'seed' : seed , 'env':env_name , 'algo' : algo , 'ndim':ndim , 'lr': learning_rate, 'target_update_freq': target_update_freq},           
                    'output_dir': OUTPUT_DIR,
                }
            )




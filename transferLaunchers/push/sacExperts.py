import os

import doodad as dd
import doodad.ec2 as ec2
import doodad.ssh as ssh
import doodad.mount as mount
from doodad.utils import EXAMPLES_DIR, REPO_DIR
import pickle


# Local docker
mode_docker = dd.mode.LocalDocker(
    image='russellm888/railrl-tf2:rllab',
)


# or this! Run experiment via docker on another machine through SSH
env_name = 'Ant'
#env_name = 'Reacher'
# or use this!
mode_ec2 = dd.mode.EC2AutoconfigDocker(
        image='russellm888/railrl-tf2:rllab',
        region='us-east-1',  # EC2 region
        # instance_type='g2.2xlarge',  # EC2 instance type
        # spot_price=0.5,  # Maximum bid price
        instance_type= 'c4.2xlarge',  # EC2 instance type
        spot_price=0.2,  # Maximum bid price
        s3_log_prefix='SACExperts',  # Folder to store log files under
        s3_log_name=env_name,
        terminate=True,  # Whether to terminate on finishing job
        
    )

MY_RUN_MODE = mode_ec2 # CHANGE THIS

# Set up code and output directories
OUTPUT_DIR = '/root/code/sac/data/'   # this is the directory visible to the target
#'/example/outputs' 


import joblib
if env_name == 'Ant':
    goals = joblib.load('/home/russellm/iclr18/rosen_data/saved_expert_traj/Expert_trajs_dense_ant/goals_pool.pkl')['goals_pool'][40:100]
    rllabDir = '~/rllab'
else :
    goals = joblib.load('/home/russellm/iclr18/rosen_data/saved_expert_traj/R7DOF/R7-ET-noise0.1-200-40-1/goals_pool.pkl')['goals_pool'][:40]
    rllabDir = '~/iclr18/maml_rosen'



mounts = [
#     mount.MountLocal(local_dir=REPO_DIR, pythonpath=True), # Code
#     mount.MountLocal(local_dir=os.path.join(EXAMPLES_DIR, 'secretlib'), pythonpath=True), # Code
# 
    mount.MountLocal(local_dir="~/doodad",
                         mount_point="/root/code/doodad",
                         filter_dir=["__pycache__", ".git"], pythonpath = True),

    mount.MountLocal(local_dir="~/sac",
                         mount_point="/root/code/sac",
                         filter_dir=["__pycache__", ".git"], pythonpath = True),

    mount.MountLocal(local_dir= rllabDir,
                         mount_point="/root/code/rllab",
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


for i, goal in enumerate(goals):

    
    expName = 'Task_' + str(i+40)
    dd.launch_python(
        target=os.path.join(THIS_FILE_DIR, '/home/russellm/sac/examples/sac_launcher.py'),
        #target=os.path.join(THIS_FILE_DIR, str(targetScript) ),  # point to a target script. If running remotely, this will be copied over
        mode=MY_RUN_MODE,
        mount_points=mounts,
        args={

            'variant' : {'log_dir' : OUTPUT_DIR+expName+'/', 'goal' : goal , 'env_name': env_name},           
            'output_dir': OUTPUT_DIR,
        }
    )
  



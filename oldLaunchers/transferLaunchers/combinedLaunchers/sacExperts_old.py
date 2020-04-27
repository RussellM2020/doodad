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


# or this! Run experiment via docker on another machine through SSH
envType = 'expertAnt' ; max_path_length = 200 ; tasksFile = 'rad2_semi' ; annotation = 'rad2_semi'

#envType = 'Push' ; tasksFile = 'push_v4' ; max_path_length = 50
#envType = 'Door' ; tasksFile = 'door_60deg' ; max_path_length = 50


mode_ec2 = dd.mode.EC2AutoconfigDocker(
        image=docker_image,
        region='us-east-1',  # EC2 region
        # instance_type='g2.2xlarge',  # EC2 instance type
        # spot_price=0.5,  # Maximum bid price
        instance_type= 'c4.2xlarge',  # EC2 instance type
        spot_price=0.4,  # Maximum bid price
        s3_log_prefix=envType+'_'+annotation,  # Folder to store log files under
        s3_log_name=envType,
        terminate=True,  # Whether to terminate on finishing job
        
    )

MY_RUN_MODE = mode_ec2 # CHANGE THIS

# Set up code and output directories
OUTPUT_DIR = '/root/code/sac/data/'   # this is the directory visible to the target
#'/example/outputs' 

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



# for seed in [0,1]:
#     for numTasks in [10,20]:
#         for rewScale in [5,10, 50, 100]:
#             for n_train in [1 ,  2 , 5, 10]:
#numTasks = 10
for seed in [0,1]:
    for n_train in [5,10]:
        for rewScale in [50, 70]:
            for reset_arg in range(3,10):
         
                expName = 'Task_'+str(reset_arg)+'/rewScale_'+str(rewScale)+'_nTrain_'+str(n_train)+'_seed_'+str(seed)
                dd.launch_python(
                    target=os.path.join(THIS_FILE_DIR, '/home/russell/sac/examples/sac_launcher.py'),
                    #target=os.path.join(THIS_FILE_DIR, str(targetScript) ),  # point to a target script. If running remotely, this will be copied over
                    mode=MY_RUN_MODE,
                    mount_points=mounts,
                    args={
                       
                        'variant' : {'log_dir' : OUTPUT_DIR+expName+'/', 'rewScale' : rewScale , 'seed' : seed , 'max_path_length': max_path_length ,\
                                 'envType':envType , 'tasksFile' : tasksFile , 'n_train': n_train , 'reset_arg':reset_arg},           
                        'output_dir': OUTPUT_DIR,
                    }
                )




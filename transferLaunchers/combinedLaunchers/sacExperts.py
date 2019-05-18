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

#envType = 'expertAnt' ; max_path_length = 200 ; tasksFile = 'rad2-semi' ; annotation = 'rad2_semi_REDONE'
#envType = 'contextualAnt' ; max_path_length = 200 ; tasksFile = 'rad2_quat_v2' ; annotation = 'quat_v2'
#envType = 'contextualPush_mug' ; max_path_length = 50 ; tasksFile = 'push_v4' ; annotation = 'v4'
#envType = 'contextualPickPlace' ; max_path_length = 50 ; tasksFile = 'push_v4' ; annotation = 'v4'
#envType = 'clawScrew' ; max_path_length = 50 ; tasksFile = 'claw_2pi' ; annotation='quat_v2'

envType = 'Push' ; tasksFile = 'push_v4' ; max_path_length = 50 ; annotation = 'lighter-mug-v4'
#envType = 'contextualDoor' ; tasksFile = 'door_60deg' ; max_path_length = 100 ; annotation = '60deg'


mode_ec2 = dd.mode.EC2AutoconfigDocker(
        image=docker_image,
        region='us-west-1',  # EC2 region
        # instance_type='g2.2xlarge',  # EC2 instance type
        # spot_price=0.5,  # Maximum bid price
        instance_type= 'c4.2xlarge',  # EC2 instance type
        spot_price=0.3,  # Maximum bid price
        s3_log_prefix=envType+'_'+annotation,  # Folder to store log files under
        s3_log_name='sac_experts',
        terminate=True,  # Whether to terminate on finishing job
        
    )

MY_RUN_MODE = mode_docker # CHANGE THIS

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

    mount.MountLocal(local_dir= "~/maml_rl",
                         mount_point="/root/code/rllab",
                         filter_dir=["__pycache__", ".git"], pythonpath = True),

    mount.MountLocal(local_dir="~/multiworld",
                         mount_point="/root/code/multiworld",
                         filter_dir=["__pycache__", ".git"], pythonpath = True),

    mount.MountLocal(local_dir="~/transferHMS",
                         mount_point="/root/code/transferHMS",
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



reset_arg = None ;
for numTasks in [10 , 20]:
    for seed in [0,1]:
        for n_train in [10]:
            for rewScale in [50]:

# n_train = 10 ; rewScale = 10 ; numTasks = 1 ; seed = 0
# for reset_arg in range(10):
                
                expName = '/rewScale_'+str(rewScale)+'_nTrain_'+str(n_train)+'/Task_'+str(reset_arg)
                dd.launch_python(
                    target=os.path.join(THIS_FILE_DIR, '/home/russell/sac/examples/sac_launcher.py'),
                    #target=os.path.join(THIS_FILE_DIR, str(targetScript) ),  # point to a target script. If running remotely, this will be copied over
                    mode=MY_RUN_MODE,
                    mount_points=mounts,
                    args={
                       
                        'variant' : {'log_dir' : OUTPUT_DIR+expName+'/', 'rewScale' : rewScale , 'seed' : seed , 'max_path_length': max_path_length ,\
                                 'envType':envType , 'tasksFile' : tasksFile , 'n_train': n_train , 'reset_arg':reset_arg , 'eval_n_episodes' : numTasks  , 'numTasks' : numTasks},           
                        'output_dir': OUTPUT_DIR,
                    }
                )





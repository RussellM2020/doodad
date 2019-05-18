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
#regionSize = '_60X20X20'
#regionSize = '20X20'
envClass = 'SawyerPusher'
#envClass = 'AntSparse'
#envClass = 'Door'
# or this! Run experiment via docker on another machine through SSH


# or use this!
mode_ec2 = dd.mode.EC2AutoconfigDocker(
        image='russellm888/railrl-tf2:rllab',
        region='us-east-1',  # EC2 region
        # instance_type='g2.2xlarge',  # EC2 instance type
        # spot_price=0.5,  # Maximum bid price
        instance_type= 'c4.8xlarge',  # EC2 instance type
        spot_price=0.6,  # Maximum bid price
        s3_log_prefix=envClass,  # Folder to store log files under
        s3_log_name='maml_gps',
        terminate=True,  # Whether to terminate on finishing job
        
    )

MY_RUN_MODE = mode_docker# CHANGE THIS

# Set up code and output directories
OUTPUT_DIR = '/root/code/maml_gps/data/'   # this is the directory visible to the target
#'/example/outputs' 
mounts = [
#     mount.MountLocal(local_dir=REPO_DIR, pythonpath=True), # Code
#     mount.MountLocal(local_dir=os.path.join(EXAMPLES_DIR, 'secretlib'), pythonpath=True), # Code
# 
    mount.MountLocal(local_dir="~/doodad",
                         mount_point="/root/code/doodad",
                         filter_dir=["__pycache__", ".git"], pythonpath = True),

    mount.MountLocal(local_dir="~/iclr18/maml_gps",
                         mount_point="/root/code/maml_gps",
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


seed = 1 ; n_parallel = 1 ; init_std = 1
#args['variant'] = {'seed' : 1, 'n_parallel' : 8 , 'log_dir': '/home/russellm/data/TRPO-SawyerPusher-New', 'envClass' : 'SawyerPusher' , 'reset_arg' : 2, 'rate': 0.01}
if envClass == 'SawyerPusher':
    max_path_length = 150
    trainGoals =  pickle.load(open('/home/russellm/multiworld/multiworld/envs/goals/PickPlace_20X20.pkl', 'rb'))[:3]
elif 'Ant' in envClass :
    max_path_length = 200

elif envClass == 'Door':
    max_path_length = 150



expert_num_itrs = 10
mbs = 3
test_on_train = True
adam_steps = 10

# targetItr = 50
# launchItr =40
#expPrefix='Target'+str(targetItr)+'_launch'+str(launchItr)+'_'
expPrefix = ''

for expert_batch_size in [10000, 20000]:
    for extra_input in  [ None]:  
        for fbs in [3, 50]:                
            for flr in [0.05, 0.01]:
                for kl_penalty in [0.05, 0, 0.1]:

                    expName = expPrefix+'fbs_'+str(fbs)+'_flr_'+str(flr)+ '_klPenalty_'+str(kl_penalty)+'_expertBatchSize_'+str(expert_batch_size)

                    dd.launch_python(
                        target=os.path.join(THIS_FILE_DIR, '/home/russellm/iclr18/maml_gps/maml_examples/maml_gps_launcher.py'),
                        #target=os.path.join(THIS_FILE_DIR, str(targetScript) ),  # point to a target script. If running remotely, this will be copied over
                        mode=MY_RUN_MODE,
                        mount_points=mounts,
                        args={

                            'variant' : {'seed' : seed, 'log_dir': OUTPUT_DIR+expName+'/', 'envClass' : envClass , 'fbs': fbs , 'mbs': mbs, 'flr': flr, 'expert_batch_size': expert_batch_size,
                                        'expert_num_itrs': expert_num_itrs ,'kl_penalty': kl_penalty,  'adam_steps': adam_steps , 'trainGoals': trainGoals, 'extra_input': extra_input},
                                         #'targetItr': targetItr , 'launchItr': launchItr},           
                            'output_dir': OUTPUT_DIR,
                        }
                    )

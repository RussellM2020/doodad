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
regionSize = ''
#envClass = 'SawyerPusher'
envClass = 'Ant'
#envClass = 'Door'
# or this! Run experiment via docker on another machine through SSH


# or use this!
mode_ec2 = dd.mode.EC2AutoconfigDocker(
        image='russellm888/railrl-tf2:rllab',
        region='us-west-1',  # EC2 region
        # instance_type='g2.2xlarge',  # EC2 instance type
        # spot_price=0.5,  # Maximum bid price
        instance_type= 'c4.2xlarge',  # EC2 instance type
        spot_price=0.3,  # Maximum bid price
        s3_log_prefix=envClass,  # Folder to store log files under
        s3_log_name='maml-il',
        terminate=True,  # Whether to terminate on finishing job
        
    )

MY_RUN_MODE = mode_docker# CHANGE THIS

# Set up code and output directories
OUTPUT_DIR = '/root/code/maml_il/data/'   # this is the directory visible to the target
#'/example/outputs' 
mounts = [
#     mount.MountLocal(local_dir=REPO_DIR, pythonpath=True), # Code
#     mount.MountLocal(local_dir=os.path.join(EXAMPLES_DIR, 'secretlib'), pythonpath=True), # Code
# 
    mount.MountLocal(local_dir="~/doodad",
                         mount_point="/root/code/doodad",
                         filter_dir=["__pycache__", ".git"], pythonpath = True),

    mount.MountLocal(local_dir="~/iclr18/maml_il",
                         mount_point="/root/code/maml_il",
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


seed = 1 ; n_parallel = 4 ; init_std = 1
#args['variant'] = {'seed' : 1, 'n_parallel' : 8 , 'log_dir': '/home/russellm/data/TRPO-SawyerPusher-New', 'envClass' : 'SawyerPusher' , 'reset_arg' : 2, 'rate': 0.01}
if envClass == 'SawyerPusher':
    max_path_length = 150
elif 'Ant' in envClass :
    max_path_length = 200

elif envClass == 'Door':
    max_path_length = 150

extra_input_dim = 10
max_pool_size = None
test_on_train = False

# targetItr = 50
# launchItr =40
#expPrefix='Target'+str(targetItr)+'_launch'+str(launchItr)+'_'
expPrefix = 'maml-il-'
sacExperts = True

for test_on_train in [True, False]:
    # for max_pool_size in [50, 20]:
        #for adam_steps in [10, 20]:
    for adam_steps in [10]:
        for fbs in [20]:
            for mbs in [20]:
                for flr in [ 0.05, 0.1,1]:

                    expName = expPrefix+'mbs_'+str(mbs)+'_fbs_'+str(fbs)+'_flr_'+str(flr)+ '_nParallel_'+str(n_parallel)+'_adamSteps_'+str(adam_steps)+\
                    '_extraInputDim_'+str(extra_input_dim)+'_initStd_'+str(init_std)+'_max_pool_size_'+str(max_pool_size)+'_testOntrain_'+str(test_on_train)

                    dd.launch_python(
                        target=os.path.join(THIS_FILE_DIR, '/home/russellm/iclr18/maml_il/maml_examples/maml_il_launcher.py'),
                        #target=os.path.join(THIS_FILE_DIR, str(targetScript) ),  # point to a target script. If running remotely, this will be copied over
                        mode=MY_RUN_MODE,
                        mount_points=mounts,
                        args={

                            'variant' : {'seed' : seed, 'n_parallel' : n_parallel , 'log_dir': OUTPUT_DIR+expName+'/', 'envClass' : envClass , 'fbs': fbs , 'mbs': mbs, 'flr': flr, 'sacExperts' : sacExperts,
                                        'adam_steps': adam_steps, 'extra_input_dim': extra_input_dim, 'init_std': init_std , 'max_pool_size' : max_pool_size , 'testOntrain': test_on_train},
                                         #'targetItr': targetItr , 'launchItr': launchItr},           
                            'output_dir': OUTPUT_DIR,
                        }
                    )
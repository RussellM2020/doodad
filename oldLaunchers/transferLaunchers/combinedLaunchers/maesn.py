import os

import doodad as dd
import doodad.ec2 as ec2
import doodad.ssh as ssh
import doodad.mount as mount
from doodad.utils import EXAMPLES_DIR, REPO_DIR
import pickle

use_gpu = False ; #docker_image = 'russellm888/railrl-gpu:tf'
#docker_image = 'russellm888/railrl-tf2:rllab'
docker_image = 'russellm888/railrl-gpu:latest'
# envType = 'Push' ; annotation = 'v4-mpl-50' ; expertDataItr = 250 ; tasksFile = 'push_v4' ; max_path_length= 50 
envType = 'Door' ; annotation = 'deg60-dense-mpl-100' ; tasksFile = 'door_60deg' ; max_path_length = 100
#envType = 'Ant' ; annotation = 'sparse-quat' ; tasksFile = 'rad2_quat' ; max_path_length = 200

mode_docker = dd.mode.LocalDocker(
    image=docker_image,
    gpu = use_gpu
)

s3_log_prefix =  'Sawyer_'+envType+'_'+annotation
s3_log_name='maesn'

if use_gpu:
    mode_ec2 = dd.mode.EC2AutoconfigDocker(
            image= docker_image,
            region='us-east-1',  # EC2 region
            instance_type='p2.xlarge',  # EC2 instance type
            spot_price=0.5,  # Maximum bid price
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
            image= docker_image ,
            region='us-west-1',  # EC2 region
            instance_type= 'c4.2xlarge',  # EC2 instance type
            spot_price=0.4,  # Maximum bid price
            s3_log_prefix = s3_log_prefix,
            s3_log_name=s3_log_name,
            terminate=True,  # Whether to terminate on finishing job
        )

MY_RUN_MODE = mode_ec2 # CHANGE THIS

# Set up code and output directories
OUTPUT_DIR = '/root/code/maesn/data/'   # this is the directory visible to the target
#'/example/outputs' 
mounts = [

    mount.MountLocal(local_dir="~/doodad",
                         mount_point="/root/code/doodad",
                         filter_dir=["__pycache__", ".git"], pythonpath = True),

    mount.MountLocal(local_dir="~/maesn_suite/maesn",
                         mount_point="/root/code/maesn",
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

init_flr = 0.5 ; n_parallel = 1 ; ldim = 2 ; kl = 0.1

for seed in range(2):
    for kl in [0.1, 0.5]:
        for fbs in [50,20]:
            for mbs in [10, 20]:
                    
                    num_total_tasks = mbs
                    expName = 'mbs_'+str(mbs)+'/ldim_'+str(ldim)+'_fbs_'+str(fbs)+'_initFlr_'+str(init_flr)+'_kl_'+str(kl)+'_seed_'+str(seed)

                    if MY_RUN_MODE == mode_docker:
                        expName = s3_log_prefix+'/'+s3_log_name+'/'+expName


                    dd.launch_python(
                        target=os.path.join(THIS_FILE_DIR, '/home/russell/maesn_suite/maesn/launchers/maesn_remote.py'),
                        #target=os.path.join(THIS_FILE_DIR, str(targetScript) ),  # point to a target script. If running remotely, this will be copied over
                        mode=MY_RUN_MODE,
                        mount_points=mounts,
                        args={
                            'variant': {'ldim':ldim, 'init_flr': init_flr, 'seed' : seed , 'log_dir':OUTPUT_DIR+expName+'/',  'n_parallel' : n_parallel, 'kl':kl,
                             'envType': envType  , 'fbs' : fbs  , 'mbs' : mbs , 'tasksFile' : tasksFile ,  'max_path_length' : max_path_length , 'num_total_tasks': num_total_tasks},
                            
                            'output_dir': OUTPUT_DIR,
                        }
                    )




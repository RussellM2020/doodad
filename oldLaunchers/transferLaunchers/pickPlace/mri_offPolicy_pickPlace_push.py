import os

import doodad as dd
import doodad.ec2 as ec2
import doodad.ssh as ssh
import doodad.mount as mount
from doodad.utils import EXAMPLES_DIR, REPO_DIR
import pickle

#russellm888/railrl-tf2
# Local docker
mode_docker = dd.mode.LocalDocker(
    image='russellm888/railrl-tf2:mri',
)


envType = 'PickPlace'
regionSize = '20X20'
# or this! Run experiment via docker on another machine through SSH


# or use this!
mode_ec2 = dd.mode.EC2AutoconfigDocker(
        image='russellm888/railrl-tf2:mri',
        region='us-west-2',  # EC2 region
        # instance_type='g2.2xlarge',  # EC2 instance type
        # spot_price=0.5,  # Maximum bid price
        instance_type= 'c4.2xlarge',  # EC2 instance type
        spot_price=0.5,  # Maximum bid price
        # s3_log_prefix='Sawyer_pickPlace_finnMAML_20X20_6_8_normalized_diffLogging',  # Folder to store log files under
       
        #s3_log_prefix='Sawyer_PickPlace_3D_block', 
        s3_log_prefix = 'Sawyer_'+envType+ '_v1',
        s3_log_name='mri_offPolicy_20Tasks_'+regionSize,
        terminate=True,  # Whether to terminate on finishing job
        
    )
#mode_ec2 = dd.mode.EC2AutoconfigDocker(
#    image='python:3.5',
#    region='us-west-1',
#    instance_type='m3.medium',
#    spot_price=0.02,
#)

MY_RUN_MODE = mode_ec2 # CHANGE THIS

# Set up code and output directories
OUTPUT_DIR = '/root/code/maml_russell/data/'   # this is the directory visible to the target
#'/example/outputs' 
mounts = [
#     mount.MountLocal(local_dir=REPO_DIR, pythonpath=True), # Code
#     mount.MountLocal(local_dir=os.path.join(EXAMPLES_DIR, 'secretlib'), pythonpath=True), # Code
# 

    mount.MountLocal(local_dir="~/doodad",
                         mount_point="/root/code/doodad",
                         filter_dir=["__pycache__", ".git"], pythonpath = True),



    mount.MountLocal(local_dir="~/maml_russell",
                         mount_point="/root/code/maml_russell",
                         filter_dir=["__pycache__", ".git"], pythonpath = True),



    mount.MountLocal(local_dir="~/multiworld",
                         mount_point="/root/code/multiworld",
                         filter_dir=["__pycache__", ".git"], pythonpath = True),

    mount.MountLocal(local_dir="~/.mujoco",
                     mount_point="/root/.mujoco",
                     filter_dir=["__pycache__", ".git"]),

    # mount.MountS3(s3_path="experiments",
    #               mount_point="/data/soroush/experiments",
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

expertDataItr = 450
expertDataLoc = '/root/code/maml_russell/saved_expert_trajs/Sawyer-'+envType+'-'+regionSize+'-v1/Itr_'+str(expertDataItr)+'/'

for seed in [0,1]:
    for init_flr in [0.05, 0.5]:
        #for ldim in [0]:
        for ldim in [2,4,8]:
            for policyType in ['fullAda_Bias' , 'biasAda_Bias' ]:
            #for policyType in ['basic_policy']:

                if policyType == 'basic_policy':
                    assert ldim == 0

                expName = 'policyType_'+policyType+'/ldim_'+str(ldim)+'_initFlr_'+str(init_flr)+'_seed_'+str(seed)
                dd.launch_python(
                    target=os.path.join(THIS_FILE_DIR, '/home/russell/maml_russell/maml_examples/remote_maml_il_sawyerPusher.py'),
                    #target=os.path.join(THIS_FILE_DIR, str(targetScript) ),  # point to a target script. If running remotely, this will be copied over
                    mode=MY_RUN_MODE,
                    mount_points=mounts,
                    args={
                        'variant': {'policyType':policyType, 'ldim':ldim, 'init_flr': init_flr, 'seed' : seed , 'log_dir':OUTPUT_DIR+expName+'/',  
                        'expertDataLoc': expertDataLoc ,  'regionSize': regionSize, 'envType': envType  },
                        
                        'output_dir': OUTPUT_DIR,
                    }
                )



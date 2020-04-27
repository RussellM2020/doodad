import os

import doodad as dd
import doodad.ec2 as ec2
import doodad.ssh as ssh
import doodad.mount as mount
from doodad.utils import EXAMPLES_DIR, REPO_DIR
import pickle


use_gpu = False
#envType = 'Push' ; annotation = '-v4-mpl-50' ; expertDataItr = 250 ; tasksFile = 'push_v4' ; max_path_length= 50 ;  expertDataLoc = '/home/code/mri/saved_expert_trajs/Sawyer-Push-v4-mpl-'+str(max_path_length)+'-numDemos5/Itr_250/'
envType = 'clawScrew' ; max_path_length = 100 ; tasksFile = 'claw_2pi' ; annotation='fixed_initPos'
expertDataLoc = '/home/code/mri/saved_expert_trajs/claw-40tasks-10demos/'
#envType = 'Door' ; annotation = 'deg60' ; expertDataItr = 150 ; expertDataLoc = '/home/code/mri/saved_expert_trajs/Sawyer-Door-deg60-mpl100-numDemos5/Itr_150/' ; tasksFile = 'door_60deg' ; max_path_length = 100


mode_docker = dd.mode.LocalDocker(
    image='russellm888/railrl-gpu:latest',
    gpu = use_gpu
)

s3_log_prefix =  'Sawyer_'+envType+'_'+annotation
s3_log_name='mri'

if use_gpu:
    mode_ec2 = dd.mode.EC2AutoconfigDocker(
            image='russellm888/railrl-gpu:tf',
            region='us-east-1',  # EC2 region
            instance_type='g3.16xlarge',  # EC2 instance type
            spot_price=1.0,  # Maximum bid price
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
            image= 'russellm888/railrl-gpu:tf' ,
            region='us-west-2',  # EC2 region
            instance_type= 'c4.2xlarge',  # EC2 instance type
            spot_price=0.4,  # Maximum bid price
            s3_log_prefix = s3_log_prefix,
            s3_log_name=s3_log_name,
            terminate=True,  # Whether to terminate on finishing job
        )

#)

MY_RUN_MODE = mode_docker# CHANGE THIS

# Set up code and output directories
OUTPUT_DIR = '/home/code/mri/data/'   # this is the directory visible to the target
#'/example/outputs' 
mounts = [
#     mount.MountLocal(local_dir=REPO_DIR, pythonpath=True), # Code
#     mount.MountLocal(local_dir=os.path.join(EXAMPLES_DIR, 'secretlib'), pythonpath=True), # Code
# 

    mount.MountLocal(local_dir="~/doodad",
                         mount_point="/home/code/doodad",
                         filter_dir=["__pycache__", ".git"], pythonpath = True),


    mount.MountLocal(local_dir="~/mri",
                         mount_point="/home/code/mri",
                         filter_dir=["__pycache__", ".git"], pythonpath = True),



    mount.MountLocal(local_dir="~/multiworld",
                         mount_point="/home/code/multiworld",
                         filter_dir=["__pycache__", ".git"], pythonpath = True),

    mount.MountLocal(local_dir="~/transferHMS",
                         mount_point="/home/code/transferHMS",
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


# mbs = 10 ;  use_maesn = False

# # for seed in [0,1]:
# #     for init_flr in [0.1, 0.5 ,1]:
# #         for fbs in [20, 50]:
# #             for policyType in ['fullAda_Bias' , 'biasAda_Bias']:
# #                 for ldim in [2,4]:

# seed = 0 ; init_flr = 0.5 ; fbs = 20 ; policyType = 'biasAda_Bias' ; ldim = 2  ; sparse = True
# adam_steps = 800
# #for adam_steps in range(800 , 2001, 200):

# expName = 'policyType_'+policyType+'/ldim_'+str(ldim)+'/adamSteps_'+str(adam_steps)+'_mbs_'+str(mbs)+'_fbs_'+str(fbs)+'_initFlr_'+str(init_flr)+'_seed_'+str(seed)

# if MY_RUN_MODE == mode_docker:
#     expName = s3_log_prefix+'/'+s3_log_name+'/'+expName


# dd.launch_python(
#     target=os.path.join(THIS_FILE_DIR, '/home/russell/mri/launchers/remote_maml_il.py'),
#     #target=os.path.join(THIS_FILE_DIR, str(targetScript) ),  # point to a target script. If running remotely, this will be copied over
#     mode=MY_RUN_MODE,
#     mount_points=mounts,
#     args={
#         'variant': {'policyType':policyType, 'ldim':ldim, 'init_flr': init_flr, 'seed' : seed , 'log_dir':OUTPUT_DIR+expName+'/',  sparse = sparse,
#         'expertDataLoc': expertDataLoc ,   'envType': envType  , 'fbs' : fbs  , 'mbs' : mbs , 'tasksFile' : tasksFile , 'adam_steps' : adam_steps , 'use_maesn' : use_maesn , 'max_path_length' : max_path_length},
        
#         'output_dir': OUTPUT_DIR,
#     }
# )


init_flr = 0.5 ;  mbs = 10 ; seed = 1 ;  ldim = 2 ; use_maesn = False


for policyType in ['biasAda_Bias' , 'fullAda_Bias' ]:
    for adam_steps in [100]:              
        for fbs in [10 , 50]: 
            expName = 'policyType_'+policyType+'/ldim_'+str(ldim)+'/adamSteps_'+str(adam_steps)+'_mbs_'+str(mbs)+'_fbs_'+str(fbs)+'_initFlr_'+str(init_flr)+'_seed_'+str(seed)

            if MY_RUN_MODE == mode_docker:
                expName = s3_log_prefix+'/'+s3_log_name+'/'+expName

            dd.launch_python(
                target=os.path.join(THIS_FILE_DIR, '/home/russell/maml_rosen/launchers/remote_maml_il.py'),
                #target=os.path.join(THIS_FILE_DIR, str(targetScript) ),  # point to a target script. If running remotely, this will be copied over
                mode=MY_RUN_MODE,
                mount_points=mounts,
                args={
                    'variant': {'policyType':policyType, 'ldim':ldim, 'init_flr': init_flr, 'seed' : seed , 'log_dir':OUTPUT_DIR+expName+'/',  
                    'expertDataLoc': expertDataLoc ,   'envType': envType  , 'fbs' : fbs  , 'mbs' : mbs , 'tasksFile' : tasksFile , 'adam_steps' : adam_steps , 'use_maesn' : use_maesn , 'max_path_length' : max_path_length},
                    
                    'output_dir': OUTPUT_DIR,
                }
            )



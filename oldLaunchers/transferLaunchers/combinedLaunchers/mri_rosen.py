import os

import doodad as dd
import doodad.ec2 as ec2
import doodad.ssh as ssh
import doodad.mount as mount
from doodad.utils import EXAMPLES_DIR, REPO_DIR
import pickle


use_gpu = False
#envType = 'clawScrew' ; max_path_length = 100 ; tasksFile = 'claw_2pi' ; annotation='fixed_initPos_mri_rosen'
#envType = 'clawScrew' ; max_path_length = 50 ; tasksFile = 'claw_2pi' ; annotation='0_2pi'
#envType = 'Door' ; max_path_length = 100 ; tasksFile = 'door_60deg' ; annotation = '60deg_dense'

#envType = 'Push' ; annotation = 'v4-mpl-50-SAC' ; tasksFile = 'push_v4' ; max_path_length = 50

#envType = 'Push' ; annotation = 'mug-v4-mpl-50' ; tasksFile = 'push_v4' ; max_path_length = 50

#envType = 'Ant' ; annotation = 'dense-semi-sanity' ; tasksFile = 'rad2_semi' ; max_path_length= 200 ; 

envType = 'Ant' ; annotation = 'dense-quat-v2-itr400' ; tasksFile = 'rad2_quat_v2' ; max_path_length= 200 ; 
#expertDataLoc = '/home/code/maml_rosen/saved_expert_trajs/claw-40tasks-10demos-equalSpacing/'
#expertDataLoc = '/home/code/maml_rosen/saved_expert_trajs/claw-sac-20tasks-1demo/'


#expertDataLoc = '/home/code/maml_rosen/saved_expert_trajs/sacAnt-10demos/'
expertDataLoc = '/home/code/maml_rosen/saved_expert_trajs/ant-quat-v2-10tasks-itr400/'
#expertDataLoc = '/home/code/maml_rosen/saved_expert_trajs/sacAnt-10demos-cutoff-450/'
#expertDataLoc = '/home/code/maml_rosen/saved_expert_trajs/trpoAnt/Itr_350/' ; annotation = 'trpo'
#expertDataLoc = '/home/code/maml_rosen/saved_expert_trajs/Ant-rad2-quat-mpl200-numDemos5/Itr_350/'
#expertDataLoc = '/home/code/maml_rosen/saved_expert_trajs/Sawyer-Push-v4-mpl-'+str(max_path_length)+'-numDemos5/Itr_250/'


#expertDataLoc = '/home/code/maml_rosen/saved_expert_trajs/Sawyer-Door-deg60-mpl100-numDemos5/Itr_150/'
#expertDataLoc = '/home/code/maml_rosen/saved_expert_trajs/SAC-pushing/'
#expertDataLoc = '/home/code/maml_rosen/saved_expert_trajs/imgObs-Push-mug-v4-mpl50-numDemos10/Itr_250/'

#expertDataLoc = '/home/code/maml_rosen/saved_expert_trajs/Push-mug-v4-mpl50-numDemos10/Itr_250/'

#expertDataLoc = '/home/code/maml_rosen/saved_expert_trajs/imgObs-Sawyer-Push-v4-mpl-'+str(max_path_length)+'-numDemos5/Itr_250/'
#
#envType = 'Door' ; annotation = 'deg60' ; expertDataItr = 150 ; expertDataLoc = '/home/code/maml_rosen/saved_expert_trajs/Sawyer-Door-deg60-mpl100-numDemos5/Itr_150/' ; tasksFile = 'door_60deg' ; max_path_length = 100


mode_docker = dd.mode.LocalDocker(
    image='russellm888/railrl-gpu:latest',
    gpu = use_gpu
)

s3_log_prefix =  'Sawyer-'+envType+'-'+annotation
s3_log_name='mri_rosen'

if use_gpu:
    s3_log_prefix +='_Images'
    mode_ec2 = dd.mode.EC2AutoconfigDocker(
            image='russellm888/railrl-gpu:latest',
            region='us-east-1',  # EC2 region
            instance_type='p2.xlarge',  # EC2 instance type
            spot_price=0.5,  # Maximum bid price
            s3_log_prefix = s3_log_prefix,
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
            image= 'russellm888/railrl-gpu:latest' ,
            region='us-west-1',  # EC2 region
            instance_type= 'c4.2xlarge',  # EC2 instance type
            spot_price=0.4,  # Maximum bid price
            s3_log_prefix = s3_log_prefix,
            s3_log_name=s3_log_name,
            terminate=True,  # Whether to terminate on finishing job
        )

#)

MY_RUN_MODE = mode_docker# CHANGE THIS

# Set up code and output directories
OUTPUT_DIR = '/home/code/maml_rosen/data/'   # this is the directory visible to the target
#'/example/outputs' 
mounts = [
#     mount.MountLocal(local_dir=REPO_DIR, pythonpath=True), # Code
#     mount.MountLocal(local_dir=os.path.join(EXAMPLES_DIR, 'secretlib'), pythonpath=True), # Code
# 
    mount.MountLocal(local_dir="~/doodad",
                         mount_point="/home/code/doodad",
                         filter_dir=["__pycache__", ".git"], pythonpath = True),


    mount.MountLocal(local_dir="~/maml_rosen",
                         mount_point="/home/code/maml_rosen",
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

use_maesn = False

# for seed in [0,1]:
#     for init_flr in [0.1, 0.5 ,1]:
#         for fbs in [20, 50]:
#             for policyType in ['fullAda_Bias' , 'biasAda_Bias']:
#                 for ldim in [2,4]:


dagger = True ; expert_policy_loc = None

init_flr = 0.5 ;  mbs = 10

for seed in [0,1]:
    for ldim in [4]:

        #for policyType in ['conv_fcBiasAda']:
        for policyType in ['fullAda_Bias']:
            for adam_steps in [500, 800 , 1000]:              
                for fbs in [20 , 50]: 
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
                            'expertDataLoc': expertDataLoc ,   'envType': envType  , 'fbs' : fbs  , 'mbs' : mbs , 'tasksFile' : tasksFile ,  'adam_steps' : adam_steps , 
                            'use_maesn' : use_maesn , 'max_path_length' : max_path_length , 'dagger': dagger , 'expert_policy_loc': expert_policy_loc , 'load_policy': None},
                            
                            'output_dir': OUTPUT_DIR,
                        }
                    )



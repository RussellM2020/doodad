import os

import doodad as dd
import doodad.ec2 as ec2
import doodad.ssh as ssh
import doodad.mount as mount
from doodad.utils import EXAMPLES_DIR, REPO_DIR
import pickle


use_gpu = False
#envType = 'clawScrew' ;  tasksFile = 'claw_2pi'
#envType = 'Coffee' ; tasksFile = 'push_v4'
expertDataPrefix = '/home/code/gmps/saved_expert_trajs/'
envType = 'SawyerMultiDomain' ; tasksFile =  'multi_domain/multiFamily_pushDoor_v1' ; annotation = 'pushDoor_v1' ; max_path_length = 100
#envType = 'SawyerMultiPush' ; max_path_length = 50 ;  annotation = 'push_2Blocks_v1' ; tasksFile = 'multi_domain/push_2Blocks_v1'
#envType = 'clawScrew' ; max_path_length = 50 ; tasksFile = 'claw_2pi' ; annotation='0_2pi'
#envType = 'Door' ; max_path_length = 100 ; tasksFile = 'door_60deg' ; annotation = '60deg_dense'
#envType = 'Push' ; annotation = 'v4-mpl-50-SAC' ; tasksFile = 'push_v4' ; max_path_length = 50
#envType = 'Push' ; annotation = 'mug-v4-mpl-50' ; tasksFile = 'push_v4' ; max_path_length = 50
#envType = 'Ant' ; annotation = 'dense-quat-v2-itr400' ; tasksFile = 'rad2_quat_v2' ; max_path_length= 200 ; 

#envType = 'Push' ; annotation = 'v4-mpl-50-SAC' ; tasksFile = 'push_v4' ; max_path_length = 50

expertDataLoc = expertDataPrefix + 'SawyerMultiDomain-pushDoor-v1-numDemos20/Itr_400/' ; max_path_length = 100

#expertDataLoc = '/home/code/gmps/saved_expert_trajs/SAC-pushing/'
#envType = 'Ant' ; annotation = 'dense-semi-sanity' ; tasksFile = 'rad2_semi' ; max_path_length= 200 ; 
#expertDataLoc = '/home/code/gmps/saved_expert_trajs/push_door/Itr_290/'
#expertDataLoc = '/home/code/gmps/saved_expert_trajs/2BlockPush-numDemos10/Itr_300/'
#expertDataLoc = '/home/code/gmps/saved_expert_trajs/pushDoor-numDemos10/Itr_400/'

#expertDataLoc = '/home/code/gmps/saved_expert_trajs/ant-quat-v2-10tasks-itr400/'
#expertDataLoc = '/home/code/gmps/saved_expert_trajs/Sawyer-Push-v4-mpl-50-numDemos5/Itr_250/' 

#expertDataLoc = '/home/code/gmps/saved_expert_trajs/Coffee-numDemos20/Itr_1/' ; max_path_length = 100 ; annotation = 'pick_place'
#expertDataLoc = '/home/code/gmps/saved_expert_trajs/SawyerMultiDomain-Push-Door-v1-mpl100-numDemos20/Itr_290/' ; max_path_length = 100 ; annotation= 'Push-Door-v1'
#saved_expert_trajs/clawScrew-v1-3tasks-rep10/' ; max_path_length = 50 ; annotation = 'quat_v1_3tasks_rep10'
#expertDataLoc = '/home/code/gmps/saved_expert_trajs/claw-40tasks-10demos-equalSpacing/' ; max_path_length = 100 ; annotation = '360_deg'


mode_docker = dd.mode.LocalDocker(
    image='russellm888/railrl-gpu:latest',
    gpu = use_gpu
)

s3_log_prefix = envType+'-'+annotation
s3_log_name='gmps_working'

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
OUTPUT_DIR = '/home/code/gmps/data/'   # this is the directory visible to the target
#'/example/outputs' 
mounts = [
#     mount.MountLocal(local_dir=REPO_DIR, pythonpath=True), # Code
#     mount.MountLocal(local_dir=os.path.join(EXAMPLES_DIR, 'secretlib'), pythonpath=True), # Code
# 
    mount.MountLocal(local_dir="~/doodad",
                         mount_point="/home/code/doodad",
                         filter_dir=["__pycache__", ".git"], pythonpath = True),


    mount.MountLocal(local_dir="~/gmps",
                         mount_point="/home/code/gmps",
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


dagger = False ; expert_policy_loc = None

init_flr = 1 ;  mbs = 3 ; diff_post_policy = False


#for init_flr in [0.001,  0.01, 0.1, 1 ]:
# for hidden_sizes in [(100,100, 100, 100), (100,100,100) , (100,100) , (200, 200, 200) , (200, 200)]:
#     for seed in [0,1]:
#         for ldim in [4, 8 , 16]:
#             for policyType in ['biasAda_Bias' , 'fullAda_Bias']:
#                 for adam_steps in [500]:              
#                     for fbs in [20 , 50]: 

hidden_sizes = (100,100)
fbs = 50 ; adam_steps = 2000; ldim = 2 ; seed = 2
policyType = 'biasAda_Bias'
#policyType = 'fullAda_Bias'
net=''
for i in hidden_sizes:
    net+=str(i)+'-'
expName = 'net_'+str(net)+'/policyType_'+policyType+'_diff_postPolicy_'+str(diff_post_policy)+'/ldim_'+str(ldim)+'/adamSteps_'+str(adam_steps)+'_mbs_'+str(mbs)+'_fbs_'+str(fbs)+'_initFlr_'+str(init_flr)+'_seed_'+str(seed)

if MY_RUN_MODE == mode_docker:
    expName = s3_log_prefix+'/'+s3_log_name+'/'+expName

dd.launch_python(
    target=os.path.join(THIS_FILE_DIR, '/home/russell/gmps/launchers/remote_train.py'),
    #target=os.path.join(THIS_FILE_DIR, str(targetScript) ),  # point to a target script. If running remotely, this will be copied over
    mode=MY_RUN_MODE,
    mount_points=mounts,
    args={
        'variant': {'policyType':policyType, 'ldim':ldim, 'init_flr': init_flr, 'seed' : seed , 'log_dir':OUTPUT_DIR+expName+'/', 'hidden_sizes' : hidden_sizes,
        'expertDataLoc': expertDataLoc ,   'envType': envType  , 'fbs' : fbs  , 'mbs' : mbs , 'tasksFile' : tasksFile ,  'adam_steps' : adam_steps , 'diff_post_policy': diff_post_policy,
        'use_maesn' : use_maesn , 'max_path_length' : max_path_length , 'dagger': dagger , 'expert_policy_loc': expert_policy_loc , 'load_policy': None},
        
        'output_dir': OUTPUT_DIR,
    }
)


#Multi-domain - Pushing best setting : fbs 50 , ldim 2 , adam 2000 , seed 2
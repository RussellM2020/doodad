import os

import doodad as dd
import doodad.ec2 as ec2
import doodad.ssh as ssh
import doodad.mount as mount
from doodad.utils import EXAMPLES_DIR, REPO_DIR
import pickle


use_gpu = False

expertDataPrefix = '/home/code/gmps_multiDom/saved_expert_trajs/'


# envType = 'SawyerMultiDomain-Push-Door-v2' ; max_path_length = 100 ; 
# tasksFile = 'multi_domain/multiDomain_pushDoorDrawer_10each' ; annotation = '_10each' ; total_num_tasks = 10
# expertDataLoc = expertDataPrefix+'SawyerMultiDomain-Push-Door-Drawer-pushDoorDrawer-10each-numDemos20/rep-Itr_400/'

# envType = 'SawyerMultiDomain-debugDoor' ; max_path_length = 100 ; 
# tasksFile = 'multi_domain/door_10each' ; annotation = '_10each' ; total_num_tasks = 10
# expertDataLoc = expertDataPrefix+'SawyerMultiDomain-debug-Door/rep-Itr_400/'


envType = 'SawyerMultiDomain-debugDrawer' ; max_path_length = 100 ; 
tasksFile = 'multi_domain/drawer_10each' ; annotation = '_10each' ; total_num_tasks = 10
expertDataLoc = expertDataPrefix+'SawyerMultiDomain-debug-Drawer/rep-Itr_400/'



mode_docker = dd.mode.LocalDocker(
    image='russellm888/railrl-gpu:latest',
    gpu = use_gpu
)

s3_log_prefix = envType+'-'+annotation
s3_log_name='gmps_multiDom_working'

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
            instance_type= 'c4.4xlarge',  # EC2 instance type
            spot_price=0.4,  # Maximum bid price
            s3_log_prefix = s3_log_prefix,
            s3_log_name=s3_log_name,
            terminate=True,  # Whether to terminate on finishing job
        )

#)

MY_RUN_MODE = mode_ec2# CHANGE THIS

# Set up code and output directories
OUTPUT_DIR = '/home/code/gmps_multiDom/data/'   # this is the directory visible to the target
#'/example/outputs' 
mounts = [
#     mount.MountLocal(local_dir=REPO_DIR, pythonpath=True), # Code
#     mount.MountLocal(local_dir=os.path.join(EXAMPLES_DIR, 'secretlib'), pythonpath=True), # Code
# 
    mount.MountLocal(local_dir="~/doodad",
                         mount_point="/home/code/doodad",
                         filter_dir=["__pycache__", ".git"], pythonpath = True),


    mount.MountLocal(local_dir="~/gmps_multiDom",
                         mount_point="/home/code/gmps_multiDom",
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

num_eval_tasks = 10
use_maesn = True

max_kl_weight = 0.01 
post_lstd = 0
diff_post_policy = False
hidden_sizes = (100,100)
policyType = 'maesn_addedNoise_v3' 
for ldim in [2,4,8]:
#for max_kl_weight in [0.01]:
    for seed in [0, 1]:
        for fbs in [20]:
            for adam_steps in [500 , 2000]:
                for init_flr in [0.5, 1, 5]:
                    for mbs in [10]:

                        net=''
                        for i in hidden_sizes:
                            net+=str(i)+'-'
                        expName = 'net_'+str(net)+'/policyType_'+policyType+'/ldim_'+str(ldim)+'/adamSteps_'+str(adam_steps)+'_mbs_'+str(mbs)+'_fbs_'+str(fbs)+'_initFlr_'+str(init_flr)+'_max_kl_'+str(max_kl_weight)+'_seed_'+str(seed)

                        if MY_RUN_MODE == mode_docker:
                            expName = s3_log_prefix+'/'+s3_log_name+'/'+expName


                        addNoise = False if 'noNoise' in policyType else True


                        dd.launch_python(
                            target=os.path.join(THIS_FILE_DIR, '/home/russell/gmps_multiDom/launchers/remote_train.py'),
                            #target=os.path.join(THIS_FILE_DIR, str(targetScript) ),  # point to a target script. If running remotely, this will be copied over
                            mode=MY_RUN_MODE,
                            mount_points=mounts,
                            args={
                                'variant': {'policyType':policyType, 'ldim':ldim, 'init_flr': init_flr, 'seed' : seed , 'log_dir':OUTPUT_DIR+expName+'/', 'num_eval_tasks' : num_eval_tasks,
                                'hidden_sizes' : hidden_sizes, 'post_lstd': post_lstd, 'addNoise' : addNoise, 'total_num_tasks' : total_num_tasks, 'max_kl_weight' : max_kl_weight,
                                'expertDataLoc': expertDataLoc ,   'envType': envType  , 'fbs' : fbs  , 'mbs' : mbs , 'tasksFile' : tasksFile ,  'adam_steps' : adam_steps , 'diff_post_policy': diff_post_policy,
                                'use_maesn' : use_maesn , 'max_path_length' : max_path_length ,  'load_policy': None},
                                
                                'output_dir': OUTPUT_DIR,
                            }
                        )

#Corr noise best settings : pushing : fbs 50 , adam 2000, init_flr 1 [peak at itr 24, 12] , ldim 2 , seed 0
#Multi-domain - Pushing best setting : fbs 50 , ldim 2 , adam 2000 , seed 2

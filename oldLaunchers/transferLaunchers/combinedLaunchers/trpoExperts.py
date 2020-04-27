import os

import doodad as dd
import doodad.ec2 as ec2
import doodad.ssh as ssh
import doodad.mount as mount
from doodad.utils import EXAMPLES_DIR, REPO_DIR
import pickle



use_gpu = False
#envType = 'Push_mug' ; annotation = 'v4-mpl-50' ;  tasksFile = 'push_v4' ; max_path_length= 50 
#envType = 'PickPlace' ; max_path_length = 50 ; tasksFile = 'push_v4' ; annotation = 'v4-mpl-50'
#envType = 'Ant' ; max_path_length = 200 ; annotation = 'quat_v2' ; tasksFile = 'rad2_quat'
#envType = 'Door' ; max_path_length = 100 ; annotation = 'deg60_handStart_4_1' ; tasksFile = 'door_60deg'
#envType = 'Claw-screw' ; max_path_length = 50 ; tasksFile = 'door_60deg' ; annotation = 'quat-v1'
#envType = 'pointMass' ; max_path_length = 100 ; tasksFile = 'door_60deg' ; annotation = 'trial'
#envType = 'Sawyer-Push' ; max_path_length = 50 ; tasksFile = 'push_v4' ; annotation = 'repeat_April'

envType = 'SawyerMultiDomain-Push-Door-Drawer' ; max_path_length = 100 ; tasksFile = 'multi_domain/multiDomain_pushDoorDrawer_20each' ; annotation = 'pushDoorDrawer_20each'
#envType = 'SawyerMultiPush' ; max_path_length = 50 ;  annotation = 'push_2Blocks_v1' ; tasksFile = 'multi_domain/push_2Blocks_v1'
#


#envType = 'Coffee' ; max_path_length = 100 ; tasksFile = 'push_v4' ; annotation = 'v1'
exp_mode = 'TRPO_individual_experts'

mode_docker = dd.mode.LocalDocker(
    image='russellm888/railrl-gpu:latest',
    gpu = use_gpu
)

s3_log_prefix =  envType+'_'+annotation
s3_log_name=exp_mode

if use_gpu:
    mode_ec2 = dd.mode.EC2AutoconfigDocker(
            image='russellm888/railrl-gpu:latest',
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
            image= 'russellm888/railrl-gpu:latest' ,
            region='us-west-2',  # EC2 region
            instance_type= 'c4.2xlarge',  # EC2 instance type
            spot_price=0.3,  # Maximum bid price
            s3_log_prefix = s3_log_prefix,
            s3_log_name=s3_log_name,
            terminate=True,  # Whether to terminate on finishing job
        )

MY_RUN_MODE = mode_ec2 # CHANGE THIS

# Set up code and output directories
OUTPUT_DIR = '/root/code/maml_rl/data/'   # this is the directory visible to the target
#'/example/outputs' 
mounts = [
#     mount.MountLocal(local_dir=REPO_DIR, pythonpath=True), # Code
#     mount.MountLocal(local_dir=os.path.join(EXAMPLES_DIR, 'secretlib'), pythonpath=True), # Code
# 
    mount.MountLocal(local_dir="~/doodad",
                         mount_point="/root/code/doodad",
                         filter_dir=["__pycache__", ".git"], pythonpath = True),

    mount.MountLocal(local_dir="~/maml_rl",
                         mount_point="/root/code/maml_rl",
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

if use_gpu:
    policyType = 'conv'
else:
    policyType = 'basic'
n_parallel = 1 ; rate = 0.01
#args['variant'] = {'seed' : 1, 'n_parallel' : 8 , 'log_dir': '/home/russellm/data/TRPO-SawyerPusher-New', 'envClass' : 'SawyerPusher' , 'reset_arg' : 2, 'rate': 0.01}
#for reset_arg in range(20):

#for seed in range(2):
#    expName = 'multiTask_seed'+str(seed)
#expName = 'Claw-screw-trial' ; seed = 0


policyType = 'basic' ; n_parallel =8 ; rate = 0.01 ; seed = 0 ; num_tasks = 30 ; batch_size = 20000

#reset_arg = None
if 'multiTask' in exp_mode:
    reset_arg = None ; num_tasks = 10
    #for reset_arg in range(20):
    for seed in [0,1]:
        for batch_size in [10000 , 200000]:
        
            expName = 'batch_size_'+str(batch_size)+'_seed_'+str(seed)
            #expName = 'Task_'+str(reset_arg)
            dd.launch_python(
                target=os.path.join(THIS_FILE_DIR, '/home/russell/maml_rl/launchers/trpo_launcher.py'),
                #target=os.path.join(THIS_FILE_DIR, str(targetScript) ),  # point to a target script. If running remotely, this will be copied over
                mode=MY_RUN_MODE,
                mount_points=mounts,
                args={

                    'variant' : {'seed' : seed, 'n_parallel' : n_parallel , 'log_dir': OUTPUT_DIR+expName+'/', 'envType' : envType , 'reset_arg' : reset_arg, 
                    'rate': rate,  'tasksFile' : tasksFile, 'max_path_length' : max_path_length , 'policyType':policyType , 'batch_size' : batch_size , 'num_tasks': num_tasks},           
                    'output_dir': OUTPUT_DIR
                }
        )

else:
    assert exp_mode == 'TRPO_individual_experts'
    #for reset_arg in range(num_tasks):
    reset_arg = 21
    expName = 'Task_'+str(reset_arg)
    dd.launch_python(
        target=os.path.join(THIS_FILE_DIR, '/home/russell/maml_rl/launchers/trpo_launcher.py'),
        #target=os.path.join(THIS_FILE_DIR, str(targetScript) ),  # point to a target script. If running remotely, this will be copied over
        mode=MY_RUN_MODE,
        mount_points=mounts,
        args={

            'variant' : {'seed' : seed, 'n_parallel' : n_parallel , 'log_dir': OUTPUT_DIR+expName+'/', 'envType' : envType , 'reset_arg' : reset_arg, 
            'rate': rate,  'tasksFile' : tasksFile, 'max_path_length' : max_path_length , 'policyType':policyType , 'batch_size' : batch_size , 'num_tasks': num_tasks},           
            'output_dir': OUTPUT_DIR
        }
    )

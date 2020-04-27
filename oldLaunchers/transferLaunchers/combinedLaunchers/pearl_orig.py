import os

import doodad as dd
import doodad.ec2 as ec2
import doodad.ssh as ssh
import doodad.mount as mount
from doodad.utils import EXAMPLES_DIR, REPO_DIR
import pickle


use_gpu = True

#envType = 'SawyerMultiDomain-Push-Door' ; annotation = '10each' ; n_train_tasks =2 ; n_test_tasks = 2 
#trainFile = 'multi_domain/multiDomain_pushDoorDrawer_10each'

envType = 'pointMass' ; annotation = 'circle-rad1' ; n_train_tasks = 20 ; n_test_tasks = 20 ; max_path_length = 200
trainFile = 'pointMass/point_circle_rad1' 
testFile =  'pointMass/point_circle_rad1'  


# envType = 'hc_vel' ; annotation = 'mean1-std1' ; n_train_tasks =20 ; n_test_tasks = 20 ; max_path_length = 200
# trainFile = 'hc_vel/hc_vel_mean1_std1_v1'
# testFile = 'hc_vel/hc_vel_mean1_std1_v2'

s3_log_prefix =  'pearl'
s3_log_name= envType+'-'+annotation 

#s3_log_name='smrl_fixed_num_tasks_true_context'
if use_gpu:
 
    mode_ec2 = dd.mode.EC2AutoconfigDocker(
            image='russellm888/smrl-gpu:latest',
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
    mode_docker = dd.mode.LocalDocker(
    image = 'russellm888/smrl-gpu:latest',
    gpu = True
    )


else:
    mode_ec2 = dd.mode.EC2AutoconfigDocker(
            image= 'russellm888/mlc:latest' ,
            region='us-west-1',  # EC2 region
            instance_type= 'c4.2xlarge',  # EC2 instance type
            spot_price=0.4,  # Maximum bid price
            s3_log_prefix = s3_log_prefix,
            s3_log_name=s3_log_name,
            terminate=True,  # Whether to terminate on finishing job
        )

    mode_docker = dd.mode.LocalDocker(
    image = 'russellm888/mlc:latest',
    gpu = False
    )

#)

MY_RUN_MODE = mode_docker# CHANGE THIS

exp_dir = '/home/code/oyster_orig/output/'
# Set up code and output directories
OUTPUT_DIR = exp_dir  # this is the directory visible to the target
#'/example/outputs' 
mounts = [

    mount.MountLocal(local_dir="~/doodad",
                         mount_point="/home/code/doodad",
                         filter_dir=["__pycache__", ".git"], pythonpath = True),


    mount.MountLocal(local_dir="~/rand_param_envs",
                         mount_point="/home/code/rand_param_envs/",
                         filter_dir=["__pycache__", ".git"], pythonpath = True),


    mount.MountLocal(local_dir="~/maml_rl",
                         mount_point="/home/code/maml_rl",
                         filter_dir=["__pycache__", ".git"], pythonpath = True),


    mount.MountLocal(local_dir="~/oyster_orig",
                         mount_point="/home/code/oyster_orig",
                         filter_dir=["__pycache__", ".git"], pythonpath = True),



    mount.MountLocal(local_dir="~/multiworld",
                         mount_point="/home/code/multiworld",
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


train = True

for seed in [0, 1, 2]:
    for uib in [True , False]:

        exp_name = 'uib_'+str(uib)+'/seed_'+str(seed)
        #env_name = envType + '_train' if train else envType + '_test'
        dd.launch_python(
          
            target=os.path.join(THIS_FILE_DIR, '/home/russell/oyster_orig/launch_experiment_custom.py'),
            mode=MY_RUN_MODE,
            mount_points=mounts,
            args={
                'variant': {'remote' : True , 'use_gpu' : use_gpu,
                             'train': train , 'seed' : seed , 'exp_name' : exp_name,
                            'envType' : envType , 'trainFile' : trainFile , 'testFile' : testFile,
                            'n_train_tasks' : n_train_tasks , 'n_test_tasks' : n_test_tasks,
                            'algo_params' :{ 'use_information_bottleneck': uib ,
                                             'max_path_length' : max_path_length}} , 
                           
                'output_dir': OUTPUT_DIR,
            }
        )
      
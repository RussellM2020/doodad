import os

import doodad as dd
import doodad.ec2 as ec2
import doodad.ssh as ssh
import doodad.mount as mount
from doodad.utils import EXAMPLES_DIR, REPO_DIR
import pickle

use_gpu = True ; #docker_image = 'russellm888/railrl-gpu:tf'
docker_image = 'russellm888/railrl-gpu:latest'
#docker_image = 'russellm888/railrl-gpu:dm'
envType = 'Push' ; annotation = 'mug-v4-mpl-50' ; expertDataItr = 250 ; tasksFile = 'push_v4' ; max_path_length= 50 
#envType = 'Door' ; annotation = 'deg60-sparse-mpl-150' ; tasksFile = 'door_60deg' ; max_path_length = 150
#envType = 'clawScrew' ; max_path_length = 100 ; tasksFile = 'claw_2pi' ; annotation='fixed_initPos'
#envType = 'Ant' ; annotation = 'dense_quat_v2_10tasks' ; tasksFile = 'quat_v2' ; max_path_length= 200 ; 

#envType = 'PickPlace' ; annotation = 'v4' ; tasksFile = 'push_v4' ; max_path_length= 50 ; 


#envType = 'pointMass' ; annotation = 'Benchmarking' ; max_path_length = 100
#envType = 'Ant' ; annotation = 'sparse-semi' ; tasksFile = 'rad2_semi' ; max_path_length = 200

mode_docker = dd.mode.LocalDocker(
	image=docker_image,
	gpu = use_gpu
)

if envType in ['Push' , 'Door' , 'PickPlace']:
	s3_log_prefix =  'Sawyer_'+envType+'_'+annotation
else:
	s3_log_prefix = envType + '_'+annotation
s3_log_name='maml-meanInfo-added'

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
			region='us-west-2',  # EC2 region
			instance_type= 'c4.2xlarge',  # EC2 instance type
			spot_price=0.35,  # Maximum bid price
			s3_log_prefix = s3_log_prefix,
			s3_log_name=s3_log_name,
			terminate=True,  # Whether to terminate on finishing job
		)

MY_RUN_MODE = mode_docker # CHANGE THIS

# Set up code and output directories
OUTPUT_DIR = '/root/code/maml_rl/data/'   # this is the directory visible to the target
#'/example/outputs' 
mounts = [

	mount.MountLocal(local_dir="~/doodad",
						 mount_point="/root/code/doodad",
						 filter_dir=["__pycache__", ".git"], pythonpath = True),

	mount.MountLocal(local_dir="~/maml_rl",
						 mount_point="/root/code/maml_rl",
						 filter_dir=["__pycache__", ".git"], pythonpath = True),



	mount.MountLocal(local_dir="~/multiworld",
						 mount_point="/root/code/multiworld",
						 filter_dir=["__pycache__", ".git"], pythonpath = True),

	# mount.MountLocal(local_dir="~/meta-rl-benchmarks",
	# 					 mount_point="/root/code/meta-rl-benchmarks",
	# 					 filter_dir=["__pycache__", ".git"], pythonpath = True),

	# mount.MountLocal(local_dir="~/transferHMS",
 #                         mount_point="/root/code/transferHMS",
 #                         filter_dir=["__pycache__", ".git"], pythonpath = True),

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

init_flr = 0.5 ; n_parallel = 1  



# policyType = 'fullAda_Bias'
# for seed in [1]:
# 	for radRange in [(0.2,  0.2) , (0.15, 0.25) , (0.1, 0.3)]:
# 		for mbs in [5, 10, 20, 40]:
# 			for fbs in [20, 50]:
				#tasksFile = 'pointMass/'+str(mbs)+'_goals_'+str(radRange[0])+'_'+str(radRange[1])+'_rad'  
				#expName = 'radRange_'+str(radRange[0])+'_'+str(radRange[1])+'/mbs_'+str(mbs)+'/'+policyType+'_ldim_'+str(ldim)+'_fbs_'+str(fbs)+'_initFlr_'+str(init_flr)+'_seed_'+str(seed)
mbs = 10 ; load_policy = None
for seed in range(2):
	for fbs in [20,50]:
		for policyType in ['conv_fcBiasAda']:
			for ldim in [2,4]:

				expName = 'policy_'+policyType+'/mbs_'+str(mbs)+'/'+policyType+'_ldim_'+str(ldim)+'_fbs_'+str(fbs)+'_initFlr_'+str(init_flr)+'_seed_'+str(seed)

				if MY_RUN_MODE == mode_docker:
					expName = s3_log_prefix+'/'+s3_log_name+'/'+expName

				dd.launch_python(
					target=os.path.join(THIS_FILE_DIR, '/home/russell/maml_rl/launchers/maml_remote_launcher.py'),
					#target=os.path.join(THIS_FILE_DIR, str(targetScript) ),  # point to a target script. If running remotely, this will be copied over
					mode=MY_RUN_MODE,
					mount_points=mounts,
					args={
						'variant': {'policyType':policyType, 'ldim':ldim, 'init_flr': init_flr, 'seed' : seed , 'log_dir':OUTPUT_DIR+expName+'/',  'n_parallel' : n_parallel,
						 'envType': envType  , 'fbs' : fbs  , 'mbs' : mbs , 'tasksFile' : tasksFile ,  'max_path_length' : max_path_length , 'load_policy': load_policy},
						
						'output_dir': OUTPUT_DIR,
					}
				)




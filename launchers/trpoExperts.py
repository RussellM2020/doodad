import os

import doodad as dd
import doodad.ec2 as ec2
import doodad.ssh as ssh
import doodad.mount as mount
from doodad.utils import EXAMPLES_DIR, REPO_DIR
import pickle

from launcher_utils import gen_mode, create_mounts , get_exp_name


# envType = 'hc_vel' ;  annotation = '' ;    num_tasks = 10 ; max_path_length = 200
# tasksFile = 'pointMass/point_circle_mean50_width10_v1'

# envType = 'pointMass' ; annotation = 'circle-mean50-width10-mpl200' ; num_tasks = 100; max_path_length = 200
# tasksFile = 'pointMass/point_circle_mean50_width10_v1' 
#testFile =  'pointMass/point_circle_mean50_width10_v2'  

# envType = 'pointMass' ; annotation = 'gear-10-debug-10-0' ; num_tasks = 20  ; max_path_length = 200
# tasksFile = 'pointMass/point_circle_rad1' 
#testFile =  'pointMass/point_circle_rad1'  
envType = 'lta-hc-disJoint' ; annotation = 'debug_0.5' ; n_train_tasks = 5 ; n_test_tasks = 1 ;  max_path_length = 200


# envType = 'SawyerMultiDomain-Push-Door-Drawer' ; max_path_length = 100 ; num_tasks = 60
# tasksFile = 'multi_domain/multiDomain_pushDoorDrawer_20each' ; annotation = 'pushDoorDrawer_20each'

exp_mode = 'TRPO_individual_experts'
#exp_mode = 'multiTask'
use_gpu = False
image = 'russellm888/railrl-gpu:latest'
log_dict = {'log_prefix': exp_mode , 'log_name':envType}

mode = 'docker'
#run_mode = gen_mode(image, use_gpu , mode = 'ssh' ,ssh_host = 'baymax')
run_mode = gen_mode(image, use_gpu , mode = mode , ssh_host = 'newton2', s3 =log_dict)
#run_mode = gen_mode(image, use_gpu , mode = 'docker')

output_dir ='/home/code/maml_rl/data/'
mounts = create_mounts(code_dirs = ['doodad' , 'maml_rl' ,  'multiworld' , 'transferHMS' , 'mlc_project/smrl',
                                    'learning_to_adapt', 'rlkit'] ,
                        output_dir = output_dir , ec2 = mode == 'ec2' )



THIS_FILE_DIR = os.path.realpath(os.path.dirname(__file__))


policyType = 'conv' if use_gpu else 'basic'

num_tasks = 6
seed = 0 ; n_parallel = 8 ; rate = 0.01
batch_size = 20000

#reset_arg = None
if 'multiTask' in exp_mode:
    reset_arg = None ; num_tasks = 10
    #for reset_arg in range(20):
    for seed in [0,1]:
        for batch_size in [10000 , 200000]:
        
            expName = get_exp_name('batch_size_'+str(batch_size)+'_seed_'+str(seed) , mode = run_mode , log_dict = log_dict)
            #expName = 'Task_'+str(reset_arg)
            dd.launch_python(
                target=os.path.join(THIS_FILE_DIR, '/home/russell/maml_rl/launchers/trpo_launcher.py'),
                #target=os.path.join(THIS_FILE_DIR, str(targetScript) ),  # point to a target script. If running remotely, this will be copied over
                mode=run_mode,
                mount_points=mounts,
                args={

                    'variant' : {'seed' : seed, 'n_parallel' : n_parallel , 'log_dir': output_dir+expName+'/', 'envType' : envType , 'reset_arg' : reset_arg, 
                    'rate': rate,  'max_path_length' : max_path_length , 'policyType':policyType ,
                    #'tasksFile' : tasksFile,
                    'batch_size' : batch_size , 'num_tasks': num_tasks},
                    'output_dir': output_dir
                }
        )

else:
    assert exp_mode == 'TRPO_individual_experts'
    for reset_arg in range(num_tasks):
    #reset_arg = 0
        expName = get_exp_name(annotation + '/Task_'+str(reset_arg) , mode = run_mode ,  log_dict = log_dict)

        dd.launch_python(
            target=os.path.join(THIS_FILE_DIR, '/home/russell/maml_rl/launchers/trpo_launcher.py'),
            #target=os.path.join(THIS_FILE_DIR, str(targetScript) ),  # point to a target script. If running remotely, this will be copied over
            mode=run_mode,
            mount_points=mounts,
            args={

                'variant' : {'seed' : seed, 'n_parallel' : n_parallel , 'log_dir': output_dir+expName+'/', 'envType' : envType , 'reset_arg' : reset_arg, 
                'rate': rate,  'max_path_length' : max_path_length , 'policyType':policyType , 'batch_size' : batch_size ,
                #'tasksFile' : tasksFile,
                'num_tasks': num_tasks},
                'output_dir': output_dir
            }
        )

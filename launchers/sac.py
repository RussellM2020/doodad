import os

import doodad as dd
import doodad.ec2 as ec2
import doodad.ssh as ssh
import doodad.mount as mount
from doodad.utils import EXAMPLES_DIR, REPO_DIR

from launcher_utils import gen_mode, create_mounts , get_exp_name
import pickle



#goal_dir= '/home/code/multiworld/multiworld/envs/goals/'
# envType = 'pointMass' ; annotation = 'circle-mean50-width10' ; n_train_tasks = 20 ; n_test_tasks = 20 ; max_path_length = 200
# trainFile = 'pointMass/point_circle_mean50_width10_v1'
# testFile =  'pointMass/point_circle_mean50_width10_v2'

envType = 'lta-hc-disJoint' ; annotation = 'debug_target_vel1_Redone' ; n_train_tasks = 6 ; max_path_length = 200

#envType = 'hc_vel' ; annotation = 'inc05_max5' ; n_train_tasks =100 ; max_path_length = 200
#trainFile = 'hc_vel/hc_vel_inc0.5_max5'

use_gpu = True
image = 'russellm888/smrl-gpu:latest'
log_dict = {'log_prefix':'sac' , 'log_name':envType}
mode = 'docker'

run_mode = gen_mode(image, use_gpu , mode = mode , ssh_host = 'newton2' , s3 = log_dict )

output_dir =  '/home/code/mlc_project/smrl/smrl/data/'
mounts = create_mounts(code_dirs = ['doodad' , 'mlc_project/smrl' , 'rlkit', 'multiworld'  , 'maml_rl' ,'learning_to_adapt'] ,
                        output_dir = output_dir , ec2 = mode == 'ec2' )

THIS_FILE_DIR = os.path.realpath(os.path.dirname(__file__))

TRAIN = True # mode 0 for train , 1 for test
device = 'cuda' if use_gpu else 'cpu'


#num_training_steps_per_epoch = 1000

load_path = '' ; load_sac_model = False

seed = 0
for num_training_steps_per_epoch in [2000]:
    for task_id in range(1):
        expName = get_exp_name(annotation+'/Task_'+str(task_id)+'/numSacSteps_'+str(num_training_steps_per_epoch), log_dict = log_dict , mode = mode)
        dd.launch_python(
            #target=os.path.join(THIS_FILE_DIR, 'smrl.rlkit_main_remote'),
            #target = 'smrl.true_context_remote',
            target = 'smrl.main',
            python_cmd = 'python -m',

            #target=os.path.join(THIS_FILE_DIR, str(targetScript) ),  # point to a target script. If running remotely, this will be copied over
            mode=run_mode,
            mount_points=mounts,
            args={
                'variant': { 'TRAIN' : TRAIN, 'device' : device, 'seed': seed , 'remote': True,

                        'train_sac_expert' : True , 'expert_task_id' : task_id,
                        'envType': envType , 'max_path_length': max_path_length,
                        #'train_task_file' : trainFile ,
                        'log_dir': output_dir+expName+'/' ,
                        'num_train_tasks': n_train_tasks ,


                        'num_training_steps_per_epoch' : num_training_steps_per_epoch ,
                        'num_epochs' : 1000,

                        'load_sac_model': load_sac_model , 'load_path':load_path
                    } ,

                'output_dir': output_dir,
            }
        )



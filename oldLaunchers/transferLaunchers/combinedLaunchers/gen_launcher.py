import os

import doodad as dd
import doodad.ec2 as ec2
import doodad.ssh as ssh
import doodad.mount as mount
from doodad.utils import EXAMPLES_DIR, REPO_DIR
import pickle


def gen_modes(image, use_gpu , mode = 'docker' , ec2_region = 'us-west-1' , s3 = {}, ssh_host = 'newton4')

    print('####################################')
    print('GPU ' , use_gpu)
    print('###################################')

    if mode == 'docker':

        return dd.mode.LocalDocker(
            image = image,
            gpu = use_gpu
        )


    elif mode == 'ssh':
        return dd.mode.SSHDocker(
                image=image,
                credentials=ssh.SSHCredentials(
                    hostname='%s.banatao.berkeley.edu' % ssh_host,
                    username='russell',
                    identity_file='~/.ssh/id_rsa'
                ),
                gpu=use_gpu
                )

    elif mode == 'ec2':

        if use_gpu:
     
            ec2_region = 'us-east-1' 
            ec2_instance_type = 'p2.xlarge'
            extra_ec2_args =  dict(
                            Placement=dict(
                                AvailabilityZone='us-east-1a',
                            ),
                        )
        else:
            ec2_instance_type = 'c4.2xlarge'
            extra_ec2_args = {}


        return  dd.mode.EC2AutoconfigDocker(
                image=image,
                region=ec2_region,  # EC2 region
                instance_type=ec2_instance_type,  # EC2 instance type
                spot_price=0.5,  # Maximum bid price
                s3_log_prefix = s3['log_prefix'],
                s3_log_name=s3['log_name'],
                gpu = use_gpu,
                terminate=True,  # Whether to terminate on finishing job
                extra_ec2_instance_kwargs = extra_ec2_args
            )

    else:
        AssertionError('Mode must be docker, ssh or ec2')

def create_mounts(code_dirs, output_dir, ec2 = False)

    mounts = [ 
            mount.MountLocal(local_dir="~/"+str(_dir),
                         mount_point="/home/code/"+str(_dir),
                         filter_dir=["__pycache__", ".git"], pythonpath = True),
            for _dir in code_dirs]

    mounts.append(mount.MountLocal(local_dir="~/.mujoco",
                     mount_point="/root/.mujoco",
                     filter_dir=["__pycache__", ".git"]),
   
    )

    if ec2:
        output_mount = mount.MountS3(s3_path='', mount_point=OUTPUT_DIR, output=True)  # use this for ec2
    else:
        output_mount = mount.MountLocal(local_dir=os.path.join(REPO_DIR, 'output'),
            mount_point=OUTPUT_DIR, output=True)
    mounts.append(output_mount)

    print(mounts)
    return mounts


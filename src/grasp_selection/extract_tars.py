import logging
import os
import sys
import tarfile
import time

import multiprocessing as mp

def run_command(command):
    try:
        print 'Running command ', command
        os.system(command)
    except:
        print 'Command %s failed' %(command) 

if __name__ == '__main__':
    argc = len(sys.argv)
    if argc < 2:
        logging.error('Must supply root directory')

    root_folder = sys.argv[1]
    tar_commands = []

    for f in os.listdir(root_folder):
        if f.endswith('.tar.gz'):
            local_file_root, ext = os.path.splitext(f)
            local_file_root, ext = os.path.splitext(local_file_root)

            filename = os.path.join(root_folder, f)
            result_dir = os.path.join(root_folder, local_file_root)

            if not os.path.exists(result_dir):
                os.mkdir(result_dir)

            command = 'sudo tar -xf %s -C %s' %(filename, result_dir)
            tar_commands.append(command)
            
    for cmd in tar_commands:
        print 'Running command', cmd
        start_time = time.time()
        os.system(cmd)
        stop_time = time.time()
        print 'Took %f sec' %(stop_time - start_time)

#    pool = mp.Pool(min(8, len(tar_commands)))
#    pool.map(run_command, tar_commands)

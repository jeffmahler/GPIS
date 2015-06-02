import os
import subprocess

FORMATS = {'.obj', '.skp', '.ply', '.off', '.3ds', '.sdf', '.db'}

BUCKET_NAME = 'dex-net'
ROOT_DIR = '/mnt/terastation/shape_data/MASTER_DB_v0'

if __name__ == '__main__':
    for dir_path, dirnames, files in os.walk(ROOT_DIR):
        for f in files:
            abs_path = os.path.join(dir_path, f)
            dataset = os.path.split(dir_path)[1]
            remote = 'gs://{}/{}'.format(BUCKET_NAME, os.path.join(dataset, f))

            format = os.path.splitext(abs_path)[1]
            if format in FORMATS:
                command = ['gsutil', 'cp', abs_path, remote]
                print(' '.join(command))
                subprocess.call(command)

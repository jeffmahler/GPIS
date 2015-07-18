import time
import subprocess

BUCKET_NAME = 'gs://dex-net/'
ROOT_DIR = '/mnt/terastation/shape_data/MASTER_DB_v1'

command = ['gsutil', '-m', 'rsync', '-r', ROOT_DIR, BUCKET_NAME]
print(' '.join(command))

start = time.time()
subprocess.call(command)
print('Uploading to {} took {} seconds.'.format(BUCKET_NAME, time.time() - start))

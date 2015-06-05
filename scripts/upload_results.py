"""
Command-line utility to upload GCE results to database.

$ python upload_results.py cm_out gs://dex-net/ \
  gs://dex-net-cm/experiment-pqfcudqamp-0.tar.gz \
  gs://dex-net-cm/experiment-pqfcudqamp-1.tar.gz
"""

import argparse
import os
import subprocess
import sys
import tarfile

TMP_DIR = '_upload_results_tmp'

def run(command):
    print ' '.join(command)
    subprocess.call(command)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('result_dir')
    parser.add_argument('database')
    parser.add_argument('results', nargs='+')
    parser.add_argument('--no-clobber', '-n', action='store_true')
    args = parser.parse_args()

    try:
        os.mkdir(TMP_DIR)
    except os.error:
        pass
    for result in results:
        name = os.path.basename(result)
        command = ['gsutil', 'cp', result, TMP_DIR]
        run(command)
        with tarfile.open(os.path.join(TMP_DIR, name)) as tar:
            tar.extractall(TMP_DIR)

    to_upload = []
    result_root = os.path.join(TMP_DIR, args.result_dir)
    for directory, directories, files in os.walk(result_root):
        to_upload.append(directory)
    to_upload.pop(0)

    for directory in to_upload:
        command = ['gsutil', '-m', 'cp', '-r', directory, database]
        if no_clobber:
            command = ['gsutil', '-m', 'cp', '-n', '-r', directory, database]
        run(command)

if __name__ == '__main__':
    main()

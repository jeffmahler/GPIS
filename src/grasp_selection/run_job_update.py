import logging
import IPython
import os
import sys

import experiment_config as ec
import job

logging.basicConfig(level=logging.INFO)
config_name = sys.argv[1]
job_name = sys.argv[2]
config = ec.ExperimentConfig(config_name)
gce_job = job.GceJob(config)
gce_job.job_name_root = job_name
gce_job.store()

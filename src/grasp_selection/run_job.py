import logging
import os
import sys

import experiment_config as ec
import job

logging.basicConfig(level=logging.INFO)
config_name = sys.argv[1]
config = ec.ExperimentConfig(config_name)
gce_job = job.GceJob(config)
gce_job.run()

# TODO: store

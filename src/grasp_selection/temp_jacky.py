import matplotlib.pyplot as plt
import numpy as np

import database as db
import experiment_config as ec

config_file = "../../cfg/test_hdf5_label_grasps_gce_jacky.yaml"
database_filename = "/mnt/terastation/shape_data/MASTER_DB_v3/dexnet_db3_01_22_16.hdf5"
dataset_name = "YCB"

config = ec.ExperimentConfig(config_file)
database = db.Hdf5Database(database_filename, config)

# read the grasp metrics and features
ds = database.dataset(dataset_name)
o = ds.object_keys
grasps = ds.grasps(o[0])
grasp_features = ds.grasp_features(o[0], grasps)
grasp_metrics = ds.grasp_metrics(o[0], grasps)

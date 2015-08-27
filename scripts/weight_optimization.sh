#! /bin/bash

GIT_DIR=/home/brian/GPIS
DATA_DIR=/home/brian/data
OUT_DIR=/home/brian/weights
mkdir $OUT_DIR

# Sleep a random amount to fend off github
MAX_TIME=$15
SLEEP_TIME=$(expr $RANDOM \% $MAX_TIME + 1)

echo 'Sleeping for' $SLEEP_TIME 'seconds'
sleep ${SLEEP_TIME}

# Update project
cd $GIT_DIR
git pull

# Mount data disk
sudo bash scripts/mount_data_disk.sh

# Retrieve metadata: dataset, chunk_start, chunk_end, bucket_name, instance_name, config
RUN_SCRIPT=$(/usr/share/google/get_metadata_value attributes/run_script)
BUCKET_NAME=$(/usr/share/google/get_metadata_value attributes/bucket_name)
INSTANCE_NAME=$(/usr/share/google/get_metadata_value attributes/instance_name)

STEP_SIZE_MAX=$(/usr/share/google/get_metadata_value attributes/step_size_max)
STEP_SIZE_PERIOD=$(/usr/share/google/get_metadata_value attributes/step_size_period)
WEIGHT_INITIAL=$(/usr/share/google/get_metadata_value attributes/weight_initial)
PARTIAL_GRADIENT_SIZE=$(/usr/share/google/get_metadata_value attributes/partial_gradient_size)

# Download config into GIT_DIR/config.yaml
CONFIG=$(/usr/share/google/get_metadata_value attributes/config)
cat <<EOF >> config.yaml
$CONFIG
step_size_max:         $STEP_SIZE_MAX
step_size_period:      $STEP_SIZE_PERIOD
weight_initial:        $WEIGHT_INITIAL
partial_gradient_size: $PARTIAL_GRADIENT_SIZE
EOF

# Run experiment
python $RUN_SCRIPT config.yaml $OUT_DIR
cp config.yaml $OUT_DIR
cd .. # back to home directory

# Zip directory and upload to bucket
cp /var/log/startupscript.log $OUT_DIR/${INSTANCE_NAME}_startupscript.log
tar -cvzf $INSTANCE_NAME.tar.gz $(basename $OUT_DIR)
sudo gsutil cp $INSTANCE_NAME.tar.gz gs://$BUCKET_NAME

# Unmount disk
sudo umount /dev/disk/by-id/google-persistent-disk-1

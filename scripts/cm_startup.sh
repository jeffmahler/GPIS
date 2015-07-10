#! /bin/bash

GIT_DIR=/home/brian/GPIS
DATA_DIR=/home/brian/data
OUT_DIR=/home/brian/cm_out
mkdir $OUT_DIR

# Update project
cd $GIT_DIR
git pull

# Mount data disk
sudo bash scripts/mount_data_disk.sh

# Retrieve metadata: dataset, chunk_start, chunk_end, bucket_name, instance_name, config
DATASET=$(curl http://metadata/computeMetadata/v1/instance/attributes/dataset -H "X-Google-Metadata-Request: True")
CHUNK_START=$(curl http://metadata/computeMetadata/v1/instance/attributes/chunk_start -H "X-Google-Metadata-Request: True")
CHUNK_END=$(curl http://metadata/computeMetadata/v1/instance/attributes/chunk_end -H "X-Google-Metadata-Request: True")
RUN_SCRIPT=$(curl http://metadata/computeMetadata/v1/instance/attributes/run_script -H "X-Google-Metadata-Request: True")
BUCKET_NAME=$(curl http://metadata/computeMetadata/v1/instance/attributes/bucket_name -H "X-Google-Metadata-Request: True")
INSTANCE_NAME=$(curl http://metadata/computeMetadata/v1/instance/attributes/instance_name -H "X-Google-Metadata-Request: True")

# Download config into GIT_DIR/config.yaml
CONFIG=$(curl http://metadata/computeMetadata/v1/instance/attributes/config -H "X-Google-Metadata-Request: True")
cat <<EOF >> config.yaml
$CONFIG
dataset:     $DATASET
chunk_start: $CHUNK_START
chunk_end:   $CHUNK_END
EOF

# Run experiment
python $RUN_SCRIPT config.yaml $OUT_DIR
cd .. # back to home directory

# Zip directory and upload to bucket
cp /var/log/startupscript.log $OUT_DIR/${INSTANCE_NAME}_startupscript.log
tar -cvzf $INSTANCE_NAME.tar.gz $(basename $OUT_DIR)
sudo gsutil cp $INSTANCE_NAME.tar.gz gs://$BUCKET_NAME

# Unmount disk
sudo umount /dev/disk/by-id/google-persistent-disk-1

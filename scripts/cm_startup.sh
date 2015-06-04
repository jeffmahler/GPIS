#! /bin/bash

GIT_DIR=/home/brian/GPIS
DATA_DIR=/home/brian/data
OUT_DIR=/home/brian/cm_out

# Update project
cd $GIT_DIR
git pull

# Mount data disk
sudo bash scripts/mount_data_disk.sh

# Retrieve metadata: bucket_name, instance_name, config
BUCKET_NAME=$(curl http://metadata/computeMetadata/v1/instance/attributes/bucket_name -H "X-Google-Metadata-Request: True")
INSTANCE_NAME=$(curl http://metadata/computeMetadata/v1/instance/attributes/instance_name -H "X-Google-Metadata-Request: True")

# Download config into GIT_DIR/config.yaml
CONFIG=$(curl http://metadata/computeMetadata/v1/instance/attributes/config -H "X-Google-Metadata-Request: True")
cat <<EOF >> config.yaml
$CONFIG
EOF

# Run experiment
python src/grasp_selection/cm_example.py config.yaml $OUT_DIR

# Zip directory and upload to bucket
cp /var/log/startupscript.log $OUT_DIR
tar -cvzf $INSTANCE_NAME.tar.gz $OUT_DIR
sudo gsutil cp $INSTANCE_NAME.tar.gz gs://$BUCKET_NAME

# Unmount disk
sudo umount /dev/disk/by-id/google-persistent-disk-1

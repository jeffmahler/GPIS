#!/bin/bash

PROJECT_DIR=/home/jmahler/abinitio-learning
CAFFE_DIR=$PROJECT_DIR/src/caffe
DATA_DIR=$CAFFE_DIR/data
LOG_FILE=$(curl http://metadata/computeMetadata/v1/instance/attributes/log_file -H "X-Google-Metadata-Request: True")

EXPERIMENT_NAME=$(curl http://metadata/computeMetadata/v1/instance/attributes/experiment_name -H "X-Google-Metadata-Request: True")
PROJECT_NAME=$(curl http://metadata/computeMetadata/v1/instance/attributes/project_name -H "X-Google-Metadata-Request: True")
BUCKET_NAME=$(curl http://metadata/computeMetadata/v1/instance/attributes/bucket_name -H "X-Google-Metadata-Request: True")

# specify image directories
IMAGE_DIR=$(curl http://metadata/computeMetadata/v1/instance/attributes/image_dir -H "X-Google-Metadata-Request: True")
IMAGE_ZIP=$IMAGE_DIR.zip
IMAGE_URL=$(curl http://metadata/computeMetadata/v1/instance/attributes/image_url -H "X-Google-Metadata-Request: True")
IMAGE_URL=$IMAGE_URL/$IMAGE_ZIP

# retrive data and unzip
cd $DATA_DIR
wget $IMAGE_URL
unzip $IMAGE_ZIP
rm $IMAGE_ZIP

# update project with git
cd $PROJECT_DIR
git pull

# copy templates to new directory and run
EXPERIMENT_DIR=$CAFFE_DIR/$IMAGE_DIR
mkdir $EXPERIMENT_DIR
cp $CAFFE_DIR/templates/* $EXPERIMENT_DIR

# copy config to directoy
CONFIG=$(curl http://metadata/computeMetadata/v1/instance/attributes/config -H "X-Google-Metadata-Request: True")
cat <<EOF >> config.yaml
$CONFIG
EOF

# hack to get python working
export PYTHONPATH=$CAFFE_DIR/python:$PYTHONPATH

# run experiment
cd $PROJECT_DIR
chmod a+x $EXPERIMENT_DIR/*
python src/train_dynamics_kalman.py -c config.yaml -o $EXPERIMENT_DIR/$LOG_FILE | tee results.log.bak

# zip directory and upload to cloud
cp /var/log/startupscript.log $CAFFE_DIR/$IMAGE_DIR
cd $CAFFE_DIR
tar cvf - $IMAGE_DIR | gzip > $EXPERIMENT_NAME.tar.gz
gsutil cp $EXPERIMENT_NAME.tar.gz gs://$BUCKET_NAME

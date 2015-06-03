#! /bin/bash
#
# Script for syncing Google Cloud Storage bucket with GCE disk.

BUCKET=gs://dex-net/
DEVICE=/dev/sdb
TARGET=/home/brian/data

# Create data directory and mount disk
mkdir -p $TARGET
sudo /usr/share/google/safe_format_and_mount -m "mkfs.ext4 -F" $DEVICE $TARGET

# Sync bucket -- https://cloud.google.com/storage/docs/gsutil/commands/rsync
sudo gsutil -m rsync -r $BUCKET $TARGET

#! /bin/bash
#
# Script for syncing Google Cloud Storage bucket with GCE disk.
# The disk must be attached in RW mode.

BUCKET=gs://dex-net/
TARGET=/home/brian/data

# Sync bucket -- https://cloud.google.com/storage/docs/gsutil/commands/rsync
sudo gsutil -m rsync -r $BUCKET $TARGET

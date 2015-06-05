#! /bin/bash
#
# Script for mounting data disk.

DEVICE=/dev/sdb
TARGET=/home/brian/data

# Create data directory and mount disk
mkdir -p $TARGET
sudo /usr/share/google/safe_format_and_mount -m "mkfs.ext4 -F" $DEVICE $TARGET

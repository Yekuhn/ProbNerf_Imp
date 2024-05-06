#!/bin/bash
DIRECTORY="/home/volt_zhou/train/glbs" # replace it with your data folder path containing glb files

for glb_file in "$DIRECTORY"/*.glb; do
  echo "Processing $glb_file"
  blender -b -P scripts/data/objaverse/blender_script.py -- --object_path "$glb_file"
done
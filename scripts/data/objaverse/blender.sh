#!/bin/bash

# Path to the JSON file containing UIDs
UID_JSON="/home/volt_zhou/train/train_uids.json"  # Adjust this to where your JSON file is located

# Base directory containing subdirectories named by UIDs
BASE_DIRECTORY="/home/volt_zhou/train/glbs"

# Read each UID from the JSON file and process .glb files in each respective directory
jq -r '.[]' "$UID_JSON" | while read uid; do
    # Construct the directory path for the current UID
    DIRECTORY="$BASE_DIRECTORY/$uid"
    
    # Check if the directory exists
    if [ -d "$DIRECTORY" ]; then
        # Process each .glb file in the directory
        for glb_file in "$DIRECTORY"/*.glb; do
            echo "Processing $glb_file"
            blender -b -P blender_script.py -- --object_path "$glb_file"
        done
    else
        echo "Directory $DIRECTORY does not exist."
    fi
done

import random
import objaverse
import gzip
import json


processes = 4

# Validation Dataset Sample Seed
random.seed(21)

uids = objaverse.load_uids()
# Validation Dataset Size
s_size = 15
random_object_uids = random.sample(uids, s_size)


# Validation Dataset Download Directory
objaverse._VERSIONED_PATH = "/home/volt_zhou/val"

valobjects = objaverse.load_objects(
    uids=random_object_uids,
    
)

    
print(valobjects)

# Training Dataset Sample Seed
random.seed(42)

# Training Dataset Sample Size
s_size = 15
random_object_uids = random.sample(uids, s_size)

# Training Dataset Download Directory
objaverse._VERSIONED_PATH = "/home/volt_zhou/train"

trainobjects = objaverse.load_objects(
    uids=random_object_uids,
    
)

    
print(trainobjects)


def decompress_json(input_path, output_path):
    with gzip.open(input_path, 'rt') as gz_file, open(output_path, 'w') as json_file:
        json.dump(json.load(gz_file), json_file, indent=4)

decompress_json('/home/volt_zhou/val/object-paths.json.gz', '/home/volt_zhou/val/object-paths.json')
decompress_json('/home/volt_zhou/train/object-paths.json.gz', '/home/volt_zhou/train/object-paths.json')

# Generate UID Json file
import os
import json

def list_folder_ids(directory_path):
    """ List all folder IDs in the specified directory. """
    return [f for f in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, f))]

def write_ids_to_json(file_path, ids):
    """ Write the list of IDs to a JSON file. """
    with open(file_path, 'w') as json_file:
        json.dump(ids, json_file, indent=4)

# Set the directory path where your folders are stored
val_path = '/home/volt_zhou/val/glbs'
train_path = '/home/volt_zhou/train/glbs'

# Set the path for the JSON file where IDs will be saved
val_path_render = '/home/volt_zhou/val/'
train_path_render = '/home/volt_zhou/train/'

# Get the list of folder IDs
val_uids = list_folder_ids(val_path)
train_uids = list_folder_ids(train_path)

# Write the folder IDs to a JSON file
write_ids_to_json(val_path_render , val_uids)
write_ids_to_json(train_path_render , train_uids)

print("Folder IDs have been written to JSON file.")


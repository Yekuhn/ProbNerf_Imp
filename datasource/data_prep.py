import random
import objaverse
import gzip
import json


processes = 4

random.seed(21)

uids = objaverse.load_uids()
random_object_uids = random.sample(uids, 10)


objaverse._VERSIONED_PATH = "/home/volt_zhou/val"
valobjects = objaverse.load_objects(
    uids=random_object_uids,
    
)

    
print(valobjects)

random.seed(42)

random_object_uids = random.sample(uids, 30)


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

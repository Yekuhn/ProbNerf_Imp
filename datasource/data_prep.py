import random
import objaverse
import gzip
import json


processes = 4

# Validation Dataset Sample Seed
random.seed(21)

uids = objaverse.load_uids()
# Validation Dataset Size
s_size = 5
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
s_size = 5
random_object_uids = random.sample(uids, s_size)

# Training Dataset Download Directory
objaverse._VERSIONED_PATH = "/home/volt_zhou/train"

trainobjects = objaverse.load_objects(
    uids=random_object_uids,
    
)

    
print(trainobjects)


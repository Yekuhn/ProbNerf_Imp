import random
import objaverse


processes = 4

random.seed(21)

uids = objaverse.load_uids()
random_object_uids = random.sample(uids, 10)


objaverse._VERSIONED_PATH = "./val"
valobjects = objaverse.load_objects(
    uids=random_object_uids,
    
)

    
print(valobjects)

random.seed(42)

random_object_uids = random.sample(uids, 30)


objaverse._VERSIONED_PATH = "./train"
trainobjects = objaverse.load_objects(
    uids=random_object_uids,
    
)

    
print(trainobjects)

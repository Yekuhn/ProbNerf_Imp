import random
import objaverse


processes = 4

random.seed(21)

uids = objaverse.load_uids()
random_object_uids = random.sample(uids, 10)



valobjects = objaverse.load_objects(
    uids=random_object_uids,
    
)

    
print(valobjects)

random.seed(42)

random_object_uids = random.sample(uids, 30)



trainobjects = objaverse.load_objects(
    uids=random_object_uids,
    
)

    
print(trainobjects)

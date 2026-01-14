from torch.utils import data
import torch
import numpy as np


# def create_mapping(entity_dir, relation_dir):
#     # entities = {}
#     # relations = {}
    
#     entities = [int(line.strip().split()[1]) for line in open(entity_dir, "r").readlines()[1:]]
#     relations = [int(line.strip().split()[1]) for line in open(relation_dir, "r").readlines()[1:]]

#     # with open(entity_dir, "r") as f:
#     #         next(f)
#     #         for line in f:
#     #             entity, entity_id = line.strip().split()
#     #             entities[int(entity_id)] = entity

#     # with open(relation_dir, "r") as f:
#     #     next(f)
#     #     for line in f:
#     #         relation, relation_id = line.strip().split()
#     #         relations[int(relation_id)] = relation   

#     return entities, relations


class DataLoader(data.Dataset):
    """Dataset implementation"""

    def __init__(self, data_dir, entity_dir, relation_dir):
        # data = []
        # with open(data_dir, "r") as f:
        #     next(f)
        #     for line in f:
        #         head, tail, relation = map(int, line.strip().split())
        #         data.append((head, relation, tail))
        
        entities = np.array([int(line.strip().split()[1]) for line in open(entity_dir, "r").readlines()[1:]])
        relations = np.array([int(line.strip().split()[1]) for line in open(relation_dir, "r").readlines()[1:]])
        
        data = torch.tensor([(int(head), int(relation), int(tail)) for head, tail, relation in (line.strip().split() for line in open(data_dir, "r").readlines()[1:])])
                
        self.entities = entities
        self.relations = relations 
        self.data = data 
        # self.data_tensor = torch.tensor(self.data, dtype=torch.long)
        
            

    def __len__(self):
        
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    
   
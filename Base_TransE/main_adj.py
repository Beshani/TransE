# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 00:22:49 2023

@author: bnwer
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 23:35:29 2023

@author: bnwer
"""

import numpy as np
import torch
import torch.nn as nn
import random
import os
import data
import time
import train_model
import torch.optim as optim
from torch.utils import data as torch_data
from tensorboardX import SummaryWriter
from collections import defaultdict
import numpy as np
#data paths
data_dir = os.path.join("Data", "Synthetic_WN18")
entity_file = os.path.join(data_dir, "entity2id.txt")
relation_file = os.path.join(data_dir, "relation2id.txt")
train_file = os.path.join(data_dir, "train2id.txt")
validation_file = os.path.join(data_dir, "valid2id.txt")
test_file = os.path.join(data_dir, "test2id.txt")
log_dir = 'logs'

#defining hyperparameters
p_batch_size = 100
n_batch_size = 50
vector_length = 50
margin = 1.0
norm = 1
learning_rate = 0.01
epochs = 100
validation_freq = 25
validation_batch_size = 50

# #entity and relation mappings
# entity, relation = data.create_mapping(entity_file, relation_file)

#create pytorch dataloaders
train_set = data.DataLoader(train_file, entity_file, relation_file)
validation_set = data.DataLoader(validation_file, entity_file, relation_file)
test_set = data.DataLoader(test_file, entity_file, relation_file)

#entity and relation mappings
entity, relation = train_set.entities, train_set.relations

train_generator = torch_data.DataLoader(train_set.data, batch_size=p_batch_size)
validation_generator = torch_data.DataLoader(validation_set.data, batch_size=validation_batch_size)
test_generator = torch_data.DataLoader(test_set.data, batch_size=validation_batch_size)

#define model

model = train_model.TransE(entity_count=len(entity), relation_count=len(relation),dim=vector_length, margin=margin, norm=norm)
model = model.to('cpu')
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
writer = SummaryWriter(log_dir)


def triple_data_mapping(data):
    unique_relations = np.unique(data[:, 1])  # Get unique relations from the second column
    data = data.numpy()
    triple_dict = {}
    for relation in unique_relations:
        relation_data = data[data[:, 1] == relation]
        heads = relation_data[:, 0]
        tails = relation_data[:, 2]
        triple_dict[relation.item()] = {'h': heads, 't':  tails}
    
    return triple_dict

def sample_corrupted_triple(positive_triple, all_entities, relation_dict):
    head, relation, tail = positive_triple.numpy()  # Convert positive_triple to NumPy array
    
    if np.random.random() < 0.5:
        true_heads = relation_dict[relation]['h']
        corrupted_head_list = np.setdiff1d(all_entities, true_heads)
        corrupted_head = np.random.choice(corrupted_head_list, size=1).item()
        corrupted_triple = np.stack([corrupted_head, relation, tail])
        
    else:
        true_tails = relation_dict[relation]['t']
        corrupted_tail_list = np.setdiff1d(all_entities, true_tails)
        corrupted_tail = np.random.choice(corrupted_tail_list, size=1).item()
        corrupted_triple = np.stack([head, relation, corrupted_tail])
    
    return corrupted_triple
    
start_time = time.time()
train_triple_dict = triple_data_mapping(train_set.data)
end_time = time.time()
print("Time taken:", end_time - start_time, "seconds")

#training loop
start_time = time.time()
for epoch in range(1, epochs+1):
    print("Starting epoch:", epoch)
    start_time_epoch = time.time()
    samples_count = 0
    final_loss = 0
    model.train() 
    
    
    for b_idx,p_batch in enumerate(train_generator):
        print(f"Starting batch: {b_idx}")
        
        start_time = time.time()
        batch_matrix = np.empty((0, 3), dtype=np.int64)
    
        for p_triple in p_batch:
           
            triple_matrix = np.expand_dims(p_triple, axis=0)
            corrupted_triples = np.array([sample_corrupted_triple(p_triple, entity, train_triple_dict) for _ in range(n_batch_size)])
            triple_matrix = np.concatenate((triple_matrix, corrupted_triples), axis=0)
            
            batch_matrix = np.concatenate((batch_matrix, triple_matrix), axis=0)
            
        end_time = time.time()
        print(f"Time taken for batch {b_idx}:", end_time - start_time, "seconds")
        
        optimizer.zero_grad()
        loss, _, _ = model.evaluate_triple(batch_matrix)
        loss.backward()
        # 
        # loss_mean.backward()
        optimizer.step()
        
        final_loss += loss.item()
        samples_count +=1
            
    average_loss = final_loss / samples_count
    print("Epoch", epoch, "Average Loss:", average_loss)
    end_time_epoch = time.time()
    epoch_time = end_time_epoch - start_time_epoch
    print("Epoch", epoch, "Time:", epoch_time, "seconds")
    writer.add_scalar('Training Loss', average_loss, epoch)
    
    # if epoch_id % validation_freq == 0:
    #     model.eval()
        


end_time = time.time()
total_time = end_time - start_time
print("Total time for all epochs:", total_time, "seconds")
writer.close()












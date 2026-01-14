import numpy as np
import torch
import torch.nn as nn
import random
import os
import data
import time
from datetime import datetime
import train_model
import torch.optim as optim
from torch.utils import data as torch_data
from tensorboardX import SummaryWriter
from collections import defaultdict
import numpy as np
import common_utils 


#data paths
data_dir = os.path.join("Data", "Synthetic_WN18")
entity_file = os.path.join(data_dir, "entity2id.txt")
relation_file = os.path.join(data_dir, "relation2id.txt")
train_file = os.path.join(data_dir, "train2id.txt")
validation_file = os.path.join(data_dir, "valid2id.txt")
test_file = os.path.join(data_dir, "test2id.txt")
checkpoint_dir =  os.path.join('checkpoints', "checkpoint.pth")
current_date_str = datetime.now().strftime('%Y-%m-%d')
log_dir = os.path.join('logs', current_date_str)
os.makedirs(log_dir, exist_ok=True)

#defining hyperparameters
p_batch_size = 100
n_batch_size = 50
vector_length = 50
margin = 1.0
norm = 1
learning_rate = 0.01
epochs = 100
validation_freq = 20
validation_batch_size = 50


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

    
# start_time = time.time()
train_triple_dict = common_utils.triple_data_mapping(train_set.data)
validation_check_data = torch.concat([train_set.data, validation_set.data], axis=0)
test_check_data = torch.concat([train_set.data, validation_set.data, test_set.data], axis=0)
# end_time = time.time()
# print("Time taken:", end_time - start_time, "seconds")



def main():
    
    #training loop
    # start_time = time.time()
    
    best_score = 0.0
    for epoch in range(1, epochs+1):
        print("Starting epoch:", epoch)
        start_time_epoch = time.time()
        samples_count = 0
        final_loss = 0
        model.train() 
        
        
        for b_idx,p_batch in enumerate(train_generator):
            print(f"Starting batch: {b_idx}")
            
            # start_time = time.time()
            batch_matrix = torch.empty(0, dtype=torch.int64)    
            for p_triple in p_batch:
                # print(p_triple)
                
                # start_time = time.time()
                triple_matrix = torch.empty([0,3],dtype=torch.int64)
                triple_matrix = torch.cat((triple_matrix, p_triple.unsqueeze(0)), dim=0)
                # triple_matrix = torch.empty([0,3],dtype=torch.int64)
                # idx = torch.arange(0,3)
                # triple_matrix = triple_matrix.index_copy_(0,idx,p_triple.unsqueeze(0))           
                for _ in range(n_batch_size):
                    corrupted_triple = common_utils.sample_corrupted_triple(p_triple, entity, train_triple_dict)
                    triple_matrix = torch.cat((triple_matrix, corrupted_triple))
                    # triple_matrix = triple_matrix.index_copy_(0,idx,corrupted_triple) 
                # end_time = time.time()
                # print("Time taken for triple:", end_time - start_time, "seconds")
                
                    
                batch_matrix = torch.cat((batch_matrix, triple_matrix.unsqueeze(0)), dim=0)
            
            # batch_matrix = torch.cat([torch.cat([p_triple.unsqueeze(0)] + [sample_corrupted_triple(p_triple, entity, train_triple_dict) for _ in range(n_batch_size)]) for p_triple in p_batch])
    
            # end_time = time.time()
            # print(f"Time taken for batch {b_idx}:", end_time - start_time, "seconds")
            
            optimizer.zero_grad()
            loss, p_distance, n_distance = model.evaluate_triple(batch_matrix)
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
        
        if epoch % validation_freq == 0:
            model.eval()
            mean_rank, hits_at_10 = common_utils.test(model=model, data_generator=validation_generator, data = validation_check_data,
                                            entity=entity,
                                            summary_writer=writer,
                                            epoch_id=epoch, metric_suffix="validation")
            
            score = mean_rank
            if score > best_score:
                best_score = score
                common_utils.save_checkpoint(model, optimizer, epoch, best_score, checkpoint_dir)
            
            


    #testing
    best_model = common_utils.load_checkpoint(checkpoint_dir)
    best_model = model.to('cpu')
    best_model.eval()
    test_mean_rank, test_hits_at_10 = common_utils.test(model=best_model, data_generator=test_generator, data = test_check_data,
                      entity=entity,
                      summary_writer=writer,
                      epoch_id=epoch, metric_suffix="test")
    
    print("Test scores: ", test_mean_rank)

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    total_time = end_time - start_time
    print("Total time for all epochs:", total_time, "seconds")
    writer.close()












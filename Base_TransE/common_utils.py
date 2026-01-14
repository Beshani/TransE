# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 01:12:13 2023

@author: bnwer
"""

import torch
import train_model
import numpy as np

def triple_data_mapping(data):
    unique_relations = np.unique(data[:, 1])  # Get unique relations from the second column
    data = data.numpy()
    triple_dict = {}
    for relation in unique_relations:
        relation_data = data[data[:, 1] == relation]
        heads = set(relation_data[:, 0])
        tails = set(relation_data[:, 2])
        triple_dict[relation.item()] = {'h': heads, 't':  tails}
    
    return triple_dict

def sample_corrupted_triple(positive_triple, all_entities, relation_dict):
    head, relation, tail = positive_triple.numpy() 
    
    true_heads = relation_dict[relation]['h']
    # corrupted_head_list = np.setdiff1d(all_entities, true_heads)
    
    true_tails = relation_dict[relation]['t']
    # corrupted_tail_list = np.setdiff1d(all_entities, true_tails)
     
    while True:
        corrupted_entity = np.random.choice(all_entities, size=1).item()        
        if np.random.random() < 0.5:
            if corrupted_entity not in true_heads: 
            # corrupted_head = np.random.choice(corrupted_head_list, size=1).item()
                corrupted_triple = np.stack([corrupted_entity, relation, tail])
                break       
        else:
            if corrupted_entity not in true_tails: 
            # corrupted_head = np.random.choice(corrupted_head_list, size=1).item()
                corrupted_triple = np.stack([head, relation, corrupted_entity])
                break       
           
            # corrupted_tail = np.random.choice(corrupted_tail_list, size=1).item()
            # corrupted_triple = np.stack([head, relation, corrupted_tail])
            
    corrupted_triple = torch.tensor(corrupted_triple).unsqueeze(0)
    # corrupted_head_list = torch.tensor(corrupted_head_list).unsqueeze(0)
    # corrupted_tail_list = torch.tensor(corrupted_tail_list).unsqueeze(0)
    
    return corrupted_triple



def hit_at_k(pred, k, true_idx):
    
    values, indices = pred.topk(k=k, largest = False)
    zero_tensor = torch.tensor([0])
    one_tensor = torch.tensor([1])
    hits = torch.where(indices == true_idx, one_tensor, zero_tensor).max().item()
    
    return hits


def load_checkpoint(checkpoint_path):
    
    checkpoint = torch.load(checkpoint_path)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['model_state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False
        
    return model

def save_checkpoint(model, optimizer, epoch_id, best_score, checkpoint_path):
    
    torch.save({
        "model": model,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch_id,
        "best_score": best_score
    }, checkpoint_path)
    
    
    
def corrupted_list(positive_triple, all_entities, relation_dict):
    head, relation, tail = positive_triple.numpy() 
    
    # true_heads = relation_dict[relation]['h']
    true_heads = np.array(list(relation_dict[relation]['h']))
    # corrupted_head_list = np.array(list(all_entities - true_heads))
    corrupted_head_list = np.setdiff1d(all_entities, true_heads)
    
    # head_matrix = np.concatenate((positive_triple.unsqueeze(0).numpy(), np.column_stack([corrupted_head_list, np.full_like(corrupted_head_list, relation), np.full_like(corrupted_head_list, tail)])))
    # corrupted_head_list =  [item for item in all_entities if item not in true_heads]
    
    # true_tails = relation_dict[relation]['t'])
    true_tails = np.array(list(relation_dict[relation]['t']))
    # corrupted_tail_list = np.array(list(all_entities - true_tails))
    corrupted_tail_list = np.setdiff1d(all_entities,true_tails)
    
    # tail_matrix = np.concatenate((positive_triple.unsqueeze(0).numpy(), np.column_stack([np.full_like(corrupted_tail_list, head), np.full_like(corrupted_tail_list, relation),corrupted_tail_list])))

    return corrupted_head_list, corrupted_tail_list
    
    
def test(model, data_generator, data, entity, summary_writer, epoch_id, metric_suffix):
         
    hits = []
    rank_sum = []
    
    # data = torch.concat([train_set.data, validation_set.data], axis=0)
    
    relation_dict = triple_data_mapping(data)
   
    for idx, batch in enumerate(data_generator):
        print(f"Starting {metric_suffix} batch: {idx}")
        
        # start_time = time.time()
       
        for p_triple in batch:
            
            head_corrupted_list, tail_corrupted_list = corrupted_list(p_triple, entity, relation_dict)
            # head_corrupted_matrix, tail_corrupted_matrix = corrupted_list(p_triple, entity, relation_dict)
            
            # head_corrupted_matrix = torch.empty([0,3],dtype=torch.int64)
            # head_corrupted_matrix = torch.cat((head_corrupted_matrix, p_triple.unsqueeze(0), head_corrupted_list), dim=0)
            # head_corrupted_matrix
            
            head_corrupted_matrix = np.tile(p_triple.numpy(), (len(head_corrupted_list)+1, 1))
            head_corrupted_matrix[1:, 0] = head_corrupted_list
        
            
            head_distance = model.calculate_distance(torch.tensor(head_corrupted_matrix).unsqueeze(0))
            # head_distance = model.calculate_distance(head_corrupted_matrix.unsqueeze(0))
            true_idx = torch.tensor(0)
            
            tail_corrupted_matrix = np.tile(p_triple.numpy(), (len(tail_corrupted_list)+1, 1))
            tail_corrupted_matrix[1:, 0] = tail_corrupted_list
            
            tail_distance = model.calculate_distance(torch.tensor(tail_corrupted_matrix).unsqueeze(0))
            # tail_distance = model.calculate_distance(tail_corrupted_matrix.unsqueeze(0))
            
            
            head_rank = (head_distance.argsort() == true_idx).nonzero().item() + 1
            tail_rank = (tail_distance.argsort() == true_idx).nonzero().item() + 1
            
            rank_sum.append(head_rank)
            rank_sum.append(tail_rank)
            
            hits.append(hit_at_k(head_distance, 10, true_idx))
            hits.append(hit_at_k(tail_distance, 10, true_idx))
            
        # end_time = time.time()
        # print(f"Time taken for batch {b_idx}:", end_time - start_time, "seconds")
    
    mr = np.mean(rank_sum)
    hits_at_10 = sum(hits)/len(hits)

    summary_writer.add_scalar('Metrics/Hits_10/' + metric_suffix, hits_at_10, global_step=epoch_id)
    summary_writer.add_scalar('Metrics/MR/' + metric_suffix, mr, global_step=epoch_id)
    
    return mr, hits_at_10



            
            
            
            
            
            
            
            
            
            
            

    
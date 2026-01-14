import numpy as np
import torch
import torch.nn as nn


class TransE(nn.Module):

    def __init__(self, entity_count, relation_count, norm=1, dim=50, margin=1):
        super(TransE, self).__init__()
        self.entity_count = entity_count
        self.relation_count = relation_count
        self.norm = norm
        self.dim = dim
        self.margin = margin
        self.entities_emb = nn.Parameter(self.init_entity_emb())
        self.relations_emb = nn.Parameter(self.init_relation_emb())
        self.loss_criterion = nn.MarginRankingLoss(margin=margin, reduction='mean')
       
    def init_entity_emb(self):
        
        emb = nn.Embedding(self.entity_count + 1, self.dim, padding_idx=self.entity_count)
        uniform_range = 6 / np.sqrt(self.dim)
        emb.weight.data.uniform_(-uniform_range, uniform_range)
        
        norm_entity_emb = torch.nn.functional.normalize(emb.weight.data, p=2,dim=1)        
        # norm = emb.weight.data.norm(p=self.norm, dim=1, keepdim = True) #p=2 euclidean
        # norm_entity_emb = emb.weight.data/norm

        return norm_entity_emb
    
    def init_relation_emb(self):
        
        emb = nn.Embedding(self.relation_count + 1, self.dim, padding_idx=self.relation_count)
        uniform_range = 6 / np.sqrt(self.dim)
        emb.weight.data.uniform_(-uniform_range, uniform_range)
        
        norm_relation_emb = torch.nn.functional.normalize(emb.weight.data, p=2,dim=1)        
        # norm = emb.weight.data.norm(p=self.norm, dim=1, keepdim = True) #p=2 euclidean
        # norm_entity_emb = emb.weight.data/norm

        return norm_relation_emb

    # def normalize_embeddings(self, entities_emb):
    #     norms = emb.weight.data.norm(p=self.norm, dim=1, keepdim=True)
    #     norm_emb = emb.weight.data/(norms)

    #     return norm_emb
    
    def calculate_distance(self, triple):
        head = triple[:,:,0]
        relations = triple[:,:,1]
        tails = triple[:,:,2]

        distance = (self.entities_emb[head] + self.relations_emb[relations] - self.entities_emb[tails])
        # distance_norm = torch.nn.functional.normalize(distance, p=self.norm, dim = 1)
        distance_norm = torch.linalg.norm(distance.squeeze(0), ord=self.norm, dim=-1)
        # distance_norm.requires_grad_(True)

        return distance_norm
    
    def calculate_loss(self, positive_distances, negative_distances):
        target = -torch.ones_like(negative_distances)
        # target.requires_grad_(True)
        
        positive_distances = torch.repeat_interleave(positive_distances, negative_distances.size()[1], 0).view(negative_distances.size())
        
        loss = self.loss_criterion(positive_distances, negative_distances, target)
        
        
        # total_loss = 0
        # for i in range(len(negative_distances)):
        #     loss = self.loss_criterion(positive_distances, negative_distances[i].unsqueeze(0), target)
        #     total_loss = total_loss + loss
    
        return loss
    
    def evaluate_triple(self, triple_matrix):
        
        positive_distances = self.calculate_distance(triple_matrix[:,0,:].unsqueeze(0))
        negative_distances = self.calculate_distance(triple_matrix[:,1:,:])
        loss = self.calculate_loss(positive_distances, negative_distances)

        return loss, positive_distances, negative_distances
    
    
    

    
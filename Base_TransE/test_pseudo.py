mean_rank_sum = 0
hits_count = 0

test_triplets = #test data

#considering removing_entity = head
for triplet in test_triplets:

    dissimilarity_scores = []
    for entity in entity_dictionary: #entity dictionary includes all the entities in our data
        corrupted_triplet  =  (entity, triplet.label(), triplet.tail())

        dissimilarity_score = pytorch_model(corrupted_triplet)  # do not compute derivatives in testing 
        dissimilarity_scores.append((entity,dissimilarity_score)) #append a tuple with head entity and corresponding score value
        

    sorted_scores = sorted(dissimilarity_scores, key=lambda x: x[1]) #sort based on dissimilarity score in the tuple 
    correct_entity = triplet.head()
    correct_entity_rank = 0

    for rank, (en, score) in enumerate(sorted_scores):
        if entity == correct_entity:
            correct_entity_rank = rank + 1
    mean_rank_sum += correct_entity_rank
    if correct_entity_rank <= 10:
            hits_count += 1


mean_rank = mean_rank_sum / len(test_triplets)
hits_at_10 = hits_count / len(test_triplets)


head_rank = []
for triplet in test_triplets:

    dissimilarity_scores = []
    corrupted_rank = 0
    for entity in entity_dictionary: #entity dictionary includes all the entities in our data
        corrupted_triplet  =  (entity, triplet.label(), triplet.tail())

        dissimilarity_score = pytorch_model(corrupted_triplet)
        if dissimilarity_score < pytorch_model(triplet):
             corrupted_rank += 1
    head_rank.append(corrupted_rank + 1)

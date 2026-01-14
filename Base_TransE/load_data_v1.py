import os


data_dir = os.path.join("Data", "WN18")


entity_file = os.path.join(data_dir, "entity2id.txt")
relation_file = os.path.join(data_dir, "relation2id.txt")
train_file = os.path.join(data_dir, "train2id.txt")
validation_file = os.path.join(data_dir, "valid2id.txt")
test_file = os.path.join(data_dir, "test2id.txt")

entities = {}
with open(entity_file, "r") as f:
    next(f)
    for line in f:
        entity, entity_id = [line[:-1].split("\t") for line in f]
        entities[int(entity_id)] = entity

relations = {}
with open(relation_file, "r") as f:
    next(f)
    for line in f:
        relation, relation_id = line.strip().split()
        relations[int(relation_id)] = relation

triples = []
with open(train_file, "r") as f:
    next(f)
    for line in f:
        head, tail, relation = map(int, line.strip().split())
        triples.append((head, tail, relation))


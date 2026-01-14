# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 03:18:23 2023

@author: bnwer
"""

import random

def read_data(file_path):
    with open(file_path, 'r') as file:
        data = [line.strip().split(',') for line in file]
    return data

def write_data(file_path, data):
    with open(file_path, 'w') as file:
        for line in data:
            file.write(','.join(map(str, line)) + '\n')

def generate_synthetic_data(original_data, percentage=0.2):
    selected_data = random.sample(original_data, int(len(original_data) * percentage))

    return selected_data

if __name__ == "__main__":
    input_file = r"C:\Users\bnwer\RIT-OneDrive\OneDrive - rit.edu\RIT\Research\1-Fall\Codes\Data\WN18\valid2id.txt"
    output_file = r"C:\Users\bnwer\RIT-OneDrive\OneDrive - rit.edu\RIT\Research\1-Fall\Codes\Data\Synthetic_WN18\valid2id.txt"
    
    original_data = read_data(input_file)
    synthetic_data = generate_synthetic_data(original_data, percentage=0.2)
    write_data(output_file, synthetic_data)

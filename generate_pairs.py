import os
import random

def generate_pairs(data_dir, output_file, num_pairs=6000):
    folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    
    same_pairs = []
    diff_pairs = []
    
    while len(same_pairs) < num_pairs // 2:
        folder = random.choice(folders)
        images = os.listdir(os.path.join(data_dir, folder))
        if len(images) < 2:
            continue
        idx1, idx2 = random.sample(range(len(images)), 2)
        same_pairs.append(f"{folder}\t{idx1}\t{idx2}")

    while len(diff_pairs) < num_pairs // 2:
        folder1, folder2 = random.sample(folders, 2)
        images1 = os.listdir(os.path.join(data_dir, folder1))
        images2 = os.listdir(os.path.join(data_dir, folder2))
        idx1 = random.randint(0, len(images1) - 1)
        idx2 = random.randint(0, len(images2) - 1)
        diff_pairs.append(f"{folder1}\t{idx1}\t{folder2}\t{idx2}")
    
    with open(output_file, 'w') as f:
        f.write(f"{num_pairs // 2}\t{num_pairs // 2}\n")
        for pair in same_pairs + diff_pairs:
            f.write(pair + "\n")

# Example usage
data_dir = 'data/output'
output_file = os.path.join(data_dir, 'pairs.txt')
generate_pairs(data_dir, output_file)

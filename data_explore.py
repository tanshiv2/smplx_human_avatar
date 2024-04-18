import pickle
import os
import numpy as np

files = ["/cluster/courses/digital_humans/datasets/team_7/data/X_Humans/00036/train/Take1/SMPLX/mesh-f00001_smplx.pkl"]

# Load the contents of the .pkl file
for file in files:
    with open(file, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

# Print keys and values
print("Keys and Values in the Pickle File:")
for key, value in data.items():
    print(f"Key: {key}")
    # print(f"Value: {value}")
    print(f"Shape:{np.array(value).shape}")



# Define your NPZ file path
npz_file_path = '/cluster/courses/digital_humans/datasets/team_7/data/ZJUMoCap/CoreView_377/models/000000.npz'  # Replace 'example.npz' with the path to your NPZ file
# npz_file_path = '/cluster/courses/digital_humans/datasets/team_7/data/X_Humans/00036/train/Take1/SMPLX/mesh-f00001_smplx.pkl'  # Replace 'example.npz' with the path to your NPZ file


# Load NPZ data
npz_data = np.load(npz_file_path)

# Get the keys of the NPZ file
# npz_keys = npz_data.keys()

# Print keys
print("Keys in the NPZ file:")
for i, (key,value) in enumerate(npz_data.items()):
    print(key)
    if i==10:
        break
import yaml
import os
import argparse

# Function to load YAML file
def load_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

# Function to save data to YAML file
def save_yaml(data, file_path):
    with open(file_path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)

# Function to edit YAML data
def edit_yaml(file_path, key, new_value):
    data = load_yaml(file_path)
    if key in data:
        data[key] = new_value
        save_yaml(data, file_path)
        print(f"Key '{key}' updated with value '{new_value}' in the YAML file.")
    else:
        print(f"Key '{key}' not found in the YAML file.")

def train_test_conf(data, file_path):
    data["dataset"]["train_subject"] = [i for i in os.listdir(os.path.join(file_path,"train"))]
    data["dataset"]["val_subject"] = [i for i in os.listdir(os.path.join(file_path,"train"))]
    data["dataset"]["test_subject"] = [i for i in os.listdir(os.path.join(file_path,"test"))]
    return data
    # print(data)

def generic_data(data_id):
    import yaml
    # Data to write to YAML file
    data = {
        '# @package': '_global_',
        'dataset_name': 'xh_' + str(data_id),
        'dataset': {
            'name': 'x_humans',
            'root_dir': '../../data/X_Humans/' + str(data_id),
            'train_subject': 'train',
            'train_views': [],
            'val_views': [],
            'test_subject': 'test',
            'test_views': [],
            'model_type': 'smplx',
            'test_frames': {
                'pose': [0, 145, 1],
                'all': [0, 145, 1]
            },
            'predict_frames': [0, 0, 1],
            'img_hw': [540, 540],
            'resolution': -1,
            'white_background': False,
            'data_device': 'cuda',
            'eval': False
        }
    }

    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate YAML file with specified parameters')
    parser.add_argument('data_id', help='Path to output YAML file')
    args = parser.parse_args()
    # Example usage
    yaml_file_path = '../configs/dataset/x_humans_' + str(args.data_id) + '.yaml'
    data_path = os.path.join('/cluster/courses/digital_humans/datasets/team_7/data/X_Humans/', str(args.data_id))
    data = generic_data(args.data_id)
    config_data = train_test_conf(data, data_path)
    save_yaml(config_data, yaml_file_path)

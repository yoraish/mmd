import torch_robotics.torch_kinematics_tree
from pathlib import Path
import yaml


# get paths
def get_root_path():
    path = Path(torch_robotics.torch_kinematics_tree.__path__[0]).resolve() / '..'
    return path


def get_data_path():
    path = get_root_path() / 'data'
    return path


def get_json_path():
    path = get_data_path() / 'jsons'
    return path


def get_imgs_path():
    path = get_data_path() / 'imgs'
    return path


def get_urdf_path():
    path = get_data_path() / 'urdf'
    return path


def get_mjcf_path():
    path = get_data_path() / 'mjcf'
    return path


def get_usd_path():
    path = get_data_path() / 'usd'
    return path


def get_robot_path():
    path = get_urdf_path() / 'robots'
    return path


def get_objects_path():
    path = get_urdf_path() / 'objects'
    return path


def get_configs_path():
    path = get_data_path() / 'configs'
    return path


def load_yaml(filename):
    with open(filename, 'r') as stream:
        try:
            config = yaml.safe_load(stream)
            print(config)
            return config
        except yaml.YAMLError as exc:
            print(exc)

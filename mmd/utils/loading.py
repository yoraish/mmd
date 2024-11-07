"""
From https://github.com/jacarvalho/mpd-public
"""
import yaml


def load_params_from_yaml(path: str):
    with open(path, "r") as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)

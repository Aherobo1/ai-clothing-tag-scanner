import json
import os
import pandas as pd

def load_tag_guides(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(__file__), '../data/tag_guides_clean.json')
    with open(path, 'r') as f:
        return json.load(f)

def load_expert_data(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(__file__), '../data/expert_data.csv')
    return pd.read_csv(path)

def load_community_data(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(__file__), '../data/community_data.csv')
    return pd.read_csv(path) 
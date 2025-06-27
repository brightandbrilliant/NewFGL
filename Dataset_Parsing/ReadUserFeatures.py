import pickle
import os
from typing import List


def read_user_features(datapath: str):
    user_features = []
    with open(datapath, 'rb') as f:
        user_features = pickle.load(f)
    return user_features


if __name__ == "__main__":
    datapath = "../dataset/dblp/user_features.pkl"
    user_features = read_user_features(datapath)
    print(len(user_features))

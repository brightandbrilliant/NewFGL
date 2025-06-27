import pickle
import os
from typing import Dict
from sentence_transformers import SentenceTransformer
import numpy as np

def read_attrs(datapath: str):
    with open(datapath, 'rb') as f:
        attrs = pickle.load(f)
    assert type(attrs) == dict
    return attrs


def parse_attrs(attrs: dict):
    attrs_list = []
    for i in range(0, len(attrs)):
        ts = attrs[i][2]
        lines = ts.strip().splitlines()
        cleaned_titles = [title.strip() for title in lines if title.strip()]
        attrs_list.append(cleaned_titles)

    return attrs_list


def sentence_to_embedding(sentences: list):
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    print('Load Baselines successfully.')
    user_features = {}
    dim = sentence_model.get_sentence_embedding_dimension()
    for i in range(0, len(sentences)):
        if len(sentences[i]) == 0:
            user_features[i] = np.zeros(dim)
            continue

        embeddings = sentence_model.encode(sentences[i], convert_to_tensor=False)
        user_features[i] = np.mean(embeddings, axis=0)
        if (i+1) % 5 == 0:
            print(user_features[i])
            print('Transforming sentences into embeddings.')

    return user_features


if __name__ == "__main__":
    datapath = '../dataset/dblp/attrs'
    attrs = read_attrs(datapath)
    attrs_list = parse_attrs(attrs)

    user_features_dict = sentence_to_embedding(attrs_list)

    output_file_path = '../dataset/dblp/user_features.pkl'
    try:
        with open(output_file_path, 'wb') as f:  # 'wb' 表示以二进制写入模式打开文件
            pickle.dump(user_features_dict, f)
        print(f"字典已成功保存到: {output_file_path}")
    except Exception as e:
        print(f"保存字典时发生错误: {e}")


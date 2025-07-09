import pickle
import os
from typing import Dict
from sentence_transformers import SentenceTransformer
import numpy as np
from Read_Stop_Word import read_stop_word
import re
import jieba

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


def clean_text(text: str, stopwords: set):
    text = re.sub(r'http\S+', '', text)  # 去掉链接
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', text)  # 只保留中英文数字
    words = jieba.lcut(text)
    filtered = [w for w in words if w not in stopwords and len(w.strip()) > 1]
    return ' '.join(filtered)


def parse_stop_word_attrs(attrs: dict, stopwords: set):
    attrs_list = []
    for i in range(len(attrs)):
        raw_text = attrs[i][2]
        lines = raw_text.strip().splitlines()
        cleaned = [clean_text(line.strip(), stopwords) for line in lines if line.strip()]
        attrs_list.append(cleaned)
    return attrs_list



if __name__ == "__main__":
    datapath = '../dataset/wd/attrs'
    attrs = read_attrs(datapath)

    for i in range(0, 5):
        print(attrs[i])

    stop_path = '../dataset/wd/stop_words_cn.pkl'
    stop_list = set(read_stop_word(stop_path))

    attrs_list = parse_stop_word_attrs(attrs, stop_list)
    for i in range(0, 5):
        print(attrs_list[i])

    user_features_dict = sentence_to_embedding(attrs_list)

    output_file_path = '../dataset/wd/user_features.pkl'
    try:
        with open(output_file_path, 'wb') as f:  # 'wb' 表示以二进制写入模式打开文件
            pickle.dump(user_features_dict, f)
        print(f"字典已成功保存到: {output_file_path}")
    except Exception as e:
        print(f"保存字典时发生错误: {e}")



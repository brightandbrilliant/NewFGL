import pickle


def read_stop_word(path):
    with open(path, 'rb') as f:
        list_ = pickle.load(f)
    return list_

if __name__ == "__main__":
    path = '../dataset/wd/stop_words_cn.pkl'
    list_ = read_stop_word(path)
    print(list_)

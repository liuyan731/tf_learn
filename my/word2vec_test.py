# -*- coding:utf-8 -*-

"""
@author: Yan Liu
@file: word2vec_test.py
@time: 2017/10/12 11:57
@desc: 参考《TensorFlow实战》Word2Vec实例
"""

import collections
import math
import os
import random
import zipfile
import numpy as np
import urllib.request
import tensorflow as tf


# 下载文本数据，并核对文件尺寸
def maybe_download(filename, expected_bytes):
    url = 'http://mattmahoney.net/dc/'
    if not os.path.exists(filename):
        filename, _ = urllib.request.urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('文件验证通过', filename)
    else:
        print('文件验证不通过...', filename)
        raise Exception('文件验证不通过...', filename)
    return filename


# 压缩文件解压，读取并处理数据（转成单词列表）
def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data


# 按照单词出现的频数创建词汇表，词频top50000的词将放入到词汇表中，top50000以外的词单独编号为0
def bulid_dataset(words):
    vocabulary_size = 50000
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary


data_index = 0  # 全局变量


# 生成Word2Vec的训练样本
def generate_batch(batch_size, num_skips, skip_window, data):
    global data_index
    assert batch_size % num_skips == 0  # 确保每个batch包含了一个词汇对应的所有样本
    assert num_skips <= 2 * skip_window  # 确保每个单词的样本数不大于单词关联其它词距离的两倍
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1
    buffer = collections.deque(maxlen=span)  # 存储一个batch_size需要处理的词和其相关的词
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)

    for i in range(batch_size // num_skips):
        target = skip_window
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels


if __name__ == '__main__':
    filename = maybe_download('text8.zip', 31344016)
    words = read_data(filename)
    print('Data size', len(words))
    data, count, dictionary, reverse_dictionary = bulid_dataset(words)
    del words
    print('Most common words (+UNK)', count[:5])
    print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])
    batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1, data=data)
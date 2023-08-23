import random
import torch
import numpy as np
import pandas as pd
import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, mean_squared_error
from torch.utils.data import Dataset, DataLoader, random_split

def generate_seq_feature2(data,
                         user_col,
                         item_col,
                         time_col,
                         item_attribute_cols=[],
                         min_item=0,
                         shuffle=True,
                         max_len=50):
    """generate sequence feature and negative sample for ranking.

    Args:
        data (pd.DataFrame): the raw data.
        user_col (str): the col name of user_id
        item_col (str): the col name of item_id
        time_col (str): the col name of timestamp
        item_attribute_cols (list[str], optional): the other attribute cols of item which you want to generate sequence feature. Defaults to `[]`.
        sample_method (int, optional): the negative sample method `{
            0: "random sampling",
            1: "popularity sampling method used in word2vec",
            2: "popularity sampling method by `log(count+1)+1e-6`",
            3: "tencent RALM sampling"}`.
            Defaults to 0.
        min_item (int, optional): the min item each user must have. Defaults to 0.
        shuffle (bool, optional): shulle if True
        max_len (int, optional): the max length of a user history sequence.

    Returns:
        pd.DataFrame: split train, val and test data with sequence features by time.
    """
    for feat in data:
        le = LabelEncoder()
        data[feat] = le.fit_transform(data[feat])
        data[feat] = data[feat].apply(lambda x: x + 1)  # 0 to be used as the symbol for padding
    data = data.astype('int32')

    # generate item to attribute mapping
    n_items = data[item_col].max()
    item2attr = {}
    if len(item_attribute_cols) > 0:
        for col in item_attribute_cols:
            map = data[[item_col, col]]
            item2attr[col] = map.set_index([item_col])[col].to_dict()

    train_data, val_data, test_data = [], [], []
    data.sort_values(time_col, inplace=True)  #sort_values按时间对数据集进行排序
    # Sliding window to construct negative samples

    for uid, hist in tqdm.tqdm(data.groupby(user_col), desc='generate sequence features'):
        pos_list = hist[item_col].tolist()
        len_pos_list = len(pos_list)
        if len_pos_list < min_item:  # drop this user when his pos items < min_item
            continue

        neg_list = [neg_sample(pos_list, n_items) for _ in range(len_pos_list)]
        for i in range(1, min(len_pos_list, max_len)):
            hist_item = pos_list[:i]
            hist_item = hist_item + [0] * (max_len - len(hist_item))
            pos_item = pos_list[i]
            neg_item = neg_list[i]
            pos_seq = [1, pos_item, uid, hist_item]
            neg_seq = [0, neg_item, uid, hist_item]
            if len(item_attribute_cols) > 0:
                for attr_col in item_attribute_cols:  # the history of item attribute features
                    hist_attr = hist[attr_col].tolist()[:i]
                    hist_attr = hist_attr + [0] * (max_len - len(hist_attr))
                    pos2attr = [hist_attr, item2attr[attr_col][pos_item]]
                    neg2attr = [hist_attr, item2attr[attr_col][neg_item]]
                    pos_seq += pos2attr
                    neg_seq += neg2attr
            if i == len_pos_list - 1:
                test_data.append(pos_seq)
                test_data.append(neg_seq)
            elif i == len_pos_list - 2:
                val_data.append(pos_seq)
                val_data.append(neg_seq)
            else:
                train_data.append(pos_seq)
                train_data.append(neg_seq)

    col_name = ['label', 'target_item_id', user_col, 'hist_item_id']
    if len(item_attribute_cols) > 0:
        for attr_col in item_attribute_cols:  # the history of item attribute features
            name = ['hist_' + attr_col, 'target_' + attr_col]
            col_name += name

    # shuffle
    if shuffle:
        random.shuffle(train_data)
        random.shuffle(val_data)
        random.shuffle(test_data)

    train = pd.DataFrame(train_data, columns=col_name)
    val = pd.DataFrame(val_data, columns=col_name)
    test = pd.DataFrame(test_data, columns=col_name)

    return train, val, test

def neg_sample(click_hist, item_size):
    neg = random.randint(1, item_size)
    while neg in click_hist:
        neg = random.randint(1, item_size)
    return neg
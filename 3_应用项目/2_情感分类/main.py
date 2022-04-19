import torch
from torch.utils.data import Dataset, DataLoader
import os, os.path
import re


def tokenlize(content):
    # 数据清洗:
    # re.sub(patten, repl) # 使用正则匹配patten, 替换为repl
    content = re.sub("<.*?>", " ", content)  # 将标签替换为空格
    __filter = ['\t', '\n', '\x97', '#', '$', '%', '&', '\.'] # 对其隐藏
    content = re.sub("|".join(__filter), " ", content)
    tokens = [word.strip().lower() for word in content.split()]

    return tokens


# 准备dataset
class IMBD_data_set(Dataset):
    def __init__(self, train=True):
        self.train_data_path = "./aclImdb_v1/aclImdb/train"
        self.test_data_path = "./aclImdb_v1/aclImdb/test"
        # 是否训练
        self.data_path = self.train_data_path if train else self.test_data_path
        self.all_file_path_list = list()
        pos_neg_path_list = os.path.join(self.data_path, "pos"), os.path.join(self.data_path, "neg")
        # self.all_file_path_list = [[os.path.join(path, file_name) for file_name in os.listdir(path) if file_name.endswith(".txt")] for path in pos_neg_path_list]
        for path in pos_neg_path_list:
            file_path_list = [os.path.join(path, file_name) for file_name in os.listdir(path) if
                              file_name.endswith(".txt")]
            self.all_file_path_list.extend(file_path_list)

    # 魔术方法之一, 将对象当成可迭代对象时, 调用的就是该方法
    def __getitem__(self, item):
        file_path = self.all_file_path_list[item]  # 此时file_path就是一个文件的路径

        label = file_path.strip().split('\\')[-2]  # 是pos还是neg
        # print(label)
        # 获取内容
        content = open(file_path).read()
        tokens = tokenlize(content)  # 分词
        # 然后应该对tokens进行编码处理.
        return tokens, label

    def __len__(self):
        return len(self.all_file_path_list)

def get_IMBD_dataloader(train=True, batch_size=2):
    return DataLoader(
        dataset=IMBD_data_set(train=True),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        # collate_fn= # 对我们的batch进行一个merges操作
                # 将batch元组通过zip函数生成可迭代对象。
    )


if __name__ == '__main__':
    for index, (x, y_true) in enumerate(get_IMBD_dataloader()):
        print(f"index{index}, x:{x}, y_true:{y_true}")
        break

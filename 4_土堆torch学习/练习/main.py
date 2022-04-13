import os.path
import cv2 as cv
import torch
from torchvision import transforms
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, train: bool = True, img_type: str = None):
        if img_type is None:
            raise RuntimeError(r"Error, param \"img_type\" is None, it should be str")
        super(MyDataset, self).__init__()
        __train_path = "./train"
        __val_path = "./val"
        # 判断是训练还是模型评估
        self.data_path = __train_path if train else __val_path  # 组合
        # print(os.listdir(self.data_path)) # ['ants_image', 'ants_label', 'bees_image', 'bees_label']
        self.image_type_folder = [folder for folder in os.listdir(self.data_path) if img_type in folder]

    def __getitem__(self, item):
        all_image_path = list()
        all_label_path = list()
        for folder in self.image_type_folder:
            if folder.endswith("image"):
                all_image_path = os.listdir(os.path.join(self.data_path, folder))
                # 注意, os.path.join()是可以添加多个属性的, 按照顺序进行拼接.
                all_image_path = [os.path.join(self.data_path, folder, file_name) for file_name in all_image_path]
            else:
                all_label_path = os.listdir(os.path.join(self.data_path, folder))
                all_label_path = [os.path.join(self.data_path, folder, file_name) for file_name in all_label_path]
                # print(all_label_path)

        content = cv.imread(all_image_path[item])
        label = open(all_label_path[item]).read()
        content = transforms.ToTensor()(content)  # 使用ToTensor进行处理.

        return content, label

    def __len__(self):
        return len(self)


if __name__ == '__main__':
    print(MyDataset(img_type="ants")[0])

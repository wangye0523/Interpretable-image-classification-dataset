"""Dataset Dataloader"""

import numpy as np
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from skimage.color import gray2rgb

CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench',
               'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
               'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
               'bowl', 'banana', 'apple', 'sandwich', 'orange',
               'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
               'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
               'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
               'cell phone', 'microwave', 'oven', 'toaster', 'sink',
               'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush')

#NUM_CLASSES = len(CLASSES) + 1


class dataset(Dataset):
    """interpretable image classification dataset with multi-granularity semantic labels"""

    def __init__(self, fine_file, coarse_file, img_dir, transform=None):
        self.images = []
        self.fine_label = []
        self.coarse_label = []

        with open(fine_file, 'r') as f:
            for line in f:
                self.images.append(line.split('\t')[0])
                self.fine_label.append(eval(line.split('\t')[1]))

        with open(coarse_file, 'r') as f1:
            for line in f1:
                self.coarse_label.append(eval(line.split('\t')[1]))

        self.transform = transform

        self.image_root_dir = img_dir

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        name = self.images[index]
        fine_label = self.fine_label[index]
        coarse_label = self.coarse_label[index]
        image_path = os.path.join(self.image_root_dir, name)


        image = torch.FloatTensor(self.load_image(path=image_path))
        gt_fine_label = self.load_fine_label(array=fine_label)
        gt_fine_label_num = self.load_fine_label_num(array=fine_label)
        gt_coarse_label = self.load_coarse_label(array=coarse_label)


        data = {

            'image': torch.FloatTensor(image),
            'fine_label': gt_fine_label,
            'fine_label_num': gt_fine_label_num,
            'coarse_label': gt_coarse_label,
        }

        return data


    def load_image(self, path=None):
        path += '.jpg'
        raw_image = Image.open(path)
        raw_image = raw_image.resize((224, 224))
        if len(np.array(raw_image).shape) != 3:
            # img = io.imread(img_path)
            raw_image = gray2rgb(np.array(raw_image))

        raw_image = np.transpose(raw_image, (2, 1, 0))
        imx_t = np.array(raw_image, dtype=np.float32) / 255.0

        return imx_t

    def load_fine_label(self, array=None):

        one_hot_label = np.zeros(80)

        for val in array:
            one_hot_label[int(val) - 1] = 1  # / len(array)
        one_hot_label = torch.from_numpy(one_hot_label)

        return one_hot_label

    def load_fine_label_num(self, array=None):

        array = torch.from_numpy(np.array(len(array)))

        return array

    def load_coarse_label(self, array=None):

        value = int(array[0]) - 1
        coarse_label = torch.tensor(value)

        return coarse_label
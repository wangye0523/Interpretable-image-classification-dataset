"""
Load dataset


Usage:
python load_dataset.py --data_root your download path

"""

from __future__ import print_function
import argparse
from dataset import dataset
import os
import torch
from torch.utils.data import DataLoader


# Arguments
parser = argparse.ArgumentParser(description='Load dataset')

parser.add_argument('--data_root', required=True)
parser.add_argument('--train_fine_path', default='train_fine_label.txt',required=False)
parser.add_argument('--train_coarse_path', default='train_coarse_label.txt',required=False)
parser.add_argument('--test_fine_path', default='test_fine_label.txt',required=False)
parser.add_argument('--test_coarse_path', default='test_coarse_label.txt',required=False)
parser.add_argument('--train_img_dir', default='dataset/train',required=False)
parser.add_argument('--test_img_dir', default='dataset/test',required=False)
parser.add_argument('--batch_size', default=32,required=False)

args = parser.parse_args()



if __name__ == "__main__":
    data_root = args.data_root
    train_fine_path = os.path.join(data_root, args.train_fine_path)
    train_coarse_path = os.path.join(data_root, args.train_coarse_path)
    test_fine_path = os.path.join(data_root, args.test_fine_path)
    test_coarse_path = os.path.join(data_root, args.test_coarse_path)
    train_img_dir = os.path.join(data_root, args.train_img_dir)
    test_img_dir = os.path.join(data_root, args.test_img_dir)


    train_dataset = dataset(fine_file=train_fine_path,
                            coarse_file=train_coarse_path,
                            img_dir=train_img_dir)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=0)

    test_dataset = dataset(fine_file=test_fine_path,
                          coarse_file=test_coarse_path,
                          img_dir=test_img_dir)

    test_dataloader = DataLoader(test_dataset,
                                batch_size=args.batch_size,
                                shuffle=False,
                                num_workers=0)

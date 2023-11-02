# data preperation creating valid set with same distribution from test part
import os
import random
import shutil
import argparse
from utils import count_plot


def data_preparation(path):
    expressions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    os.chdir(path)
    if os.path.isdir('valid/') is False:
        os.mkdir('valid')

    for expression in expressions:
        # val_f = os.path.join(path + f'test/{expression}')
        img_lst = os.listdir(f'test/{expression}')
        dist = int(len(img_lst) * 0.7)
        test_samples = random.sample(img_lst,dist)
        os.mkdir(f'valid/{expression}')
        for k in test_samples:
            shutil.move(f'test/{expression}/{k}',f'valid/{expression}')
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Data Preperation, creating validation")
    parser.add_argument("--dataset_path", required=True, help="Path to the dataset")
    args = parser.parse_args()
    data_preparation(args.dataset_path)
    count_plot((os.path.join(args.dataset_path, 'validation')),'valid_count')
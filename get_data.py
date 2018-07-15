import numpy as np
import cv2
import os

loc_src = "/home/muhsin/code/master_thes/prepare_data/data/img_preprocessed_set_1/"

def getClassNum(video_file_name):
    """does some string operations to obtain class num"""
    file_parts = video_file_name.split('_')
    class_of_file = int(file_parts[0])
    return class_of_file

def readImg(file_name, loc_src=loc_src):
    file_loc = loc_src + file_name
    img = cv2.imread(file_loc)
    img = cv2.resize(img, (32, 32))
    return img

def load_data():
    file_names = os.listdir(loc_src)
    X, y = [], []
    for file in file_names:
        y.append(getClassNum(file))
        img = readImg(file)
        X.append(img[:,:,0])
    return X, y

def main():
    # X, y = load_data()
    pass

if __name__ == '__main__':
    main()

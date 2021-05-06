import os
import sys
import random
import shutil

#从训练集中每个类别中抽取6张作为测试集
def copyFile(fileDir):
    pathDir = os.listdir(fileDir)
    sample = random.sample(pathDir, 6)
    print(sample)
    for name in sample:
        shutil.copyfile(fileDir + name, tarDir + name)
 
if __name__ == '__main__':
    # open /textiles
    path = "./dataset/VeRi_prepare/train_all1"
    dirs = os.listdir(path)
    i = 0
    # output all folds
    for file in dirs:
        print(file)
        i = i+1
        filename = "./dataset/VeRi_prepare/test1/" + str(file)
        os.makedirs(filename)
        fileDir = path + "/" + str(file) + "/"
        tarDir = filename + "/"
        print("fileDir:", fileDir)
        print("tarDir:", tarDir)
        copyFile(fileDir)

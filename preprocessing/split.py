#
import os
import shutil


def ensure_path(path):
    if not (os.path.exists(path)):
        os.makedirs(path)

def split_npy_txt(path_list):
    npy_list = [p for p in path_list if p.endswith('.npy')]
    txt_list = [p for p in path_list if p.endswith('.txt')]
    return npy_list, txt_list

def split_data(source, target, ratio = 0.5):
    dir_path_s = list(map(lambda x: os.path.join(source, x), os.listdir(source)))
    dir_path_t = list(map(lambda x: os.path.join(target, x), os.listdir(source)))
    # print(dir_path)
    for i in dir_path_t:
        ensure_path(i)
    num_dirs = len(dir_path_s)
    for p in range(num_dirs):
        file_path_s = list(map(lambda x: os.path.join(dir_path_s[p], x), os.listdir(dir_path_s[p])))
        # print(os.listdir(p))
        # file_path_t = list(map(lambda x: os.path.join(target, x), os.listdir(p)))
        npy_path_s, txt_path_s = split_npy_txt(file_path_s)
        # npy_path_t, txt_path_t = split_npy_txt(file_path_t)
        num_dev = int(len(npy_path_s) * ratio)
        # print(num_dev)
        for i in range(num_dev):
            shutil.move(npy_path_s[i], dir_path_t[p])
            shutil.move(txt_path_s[i], dir_path_t[p])


if __name__ == '__main__':
    source_path = ".\\corpus\\test"
    target_path = ".\\corpus\\dev"
    split_data(source_path, target_path)

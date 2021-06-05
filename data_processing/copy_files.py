import argparse
from shutil import copy2



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file-list-name', type=str, default='')
    parser.add_argument('--target-directory', type=str, default='~/user/')
    args = parser.parse_args()

    f = open(args.file_list_name, 'r')
    file_list = [i.split() for i in f.readlines().strip()]
    f.close()

    for i in file_list:
        for j in i:
            copy2(j, args.target_directory)

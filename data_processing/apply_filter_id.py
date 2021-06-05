if __name__ == '__main__':
    f = open('test_file_list.txt', 'r')
    train_file_list = f.readlines()
    f.close()

    f = open('matterport_eval_keep_id.txt', 'r')
    ids = [int(i) for i in f.readlines()]
    ids = set(ids)
    f.close()

    f = open('filtered_test_file_list_125.txt', 'w+')
    for i in range(len(train_file_list)):
        if i in ids:
            f.write(train_file_list[i])
    f.close()

import os, re

# TEST
# ROOT_DIRECTORY = r'/Users/user/Downloads'
# Train
ROOT_DIRECTORY = r'/usr/xtmp/user'

COLOR_ROOT = os.path.join(ROOT_DIRECTORY, 'matterport_undistorted_color')
DEPTH_ROOT = os.path.join(ROOT_DIRECTORY, 'matterport_undistorted_depth')
RENDER_DEPTH_ROOT = os.path.join(ROOT_DIRECTORY, 'matterport_render_depth')


def write_file_list(filename, file_list):
    f = open(filename, 'w+')
    for i in file_list:
        f.write(' '.join(i))
        f.write('\n')
    f.close()

f = open('../tmp/test_list_temp.txt', 'r')
test_file_list = [i.strip() for i in f.readlines()]
f.close()

test_file_paths = []

for i in test_file_list:
    sequence_id, _, color_name = i.split(r'/')
    segments = re.split(r'[_.]', color_name)

    depth_name = segments[0] + '_' + segments[1].replace('i', 'd') + '_' + segments[2] + '.png'
    render_depth_name = segments[0] + '_' + segments[1].replace('i', 'd') + '_' + segments[
        2] + '_mesh_depth.png'

    color_path = os.path.join(COLOR_ROOT, sequence_id, color_name)
    depth_path = os.path.join(DEPTH_ROOT, sequence_id, depth_name)
    render_depth_path = os.path.join(RENDER_DEPTH_ROOT, sequence_id, 'mesh_images', render_depth_name)

    entry = (color_path, depth_path, render_depth_path)

    test_file_paths.append(entry)

write_file_list('test_file_list.txt', test_file_paths)

f = open('../tmp/train_list_temp.txt', 'r')
test_file_list = [i.strip() for i in f.readlines()]
f.close()

train_file_paths = []

for i in test_file_list:
    sequence_id, _, color_name = i.split(r'/')
    segments = re.split(r'[_.]', color_name)

    depth_name = segments[0] + '_' + segments[1].replace('i', 'd') + '_' + segments[2] + '.png'
    render_depth_name = segments[0] + '_' + segments[1].replace('i', 'd') + '_' + segments[
        2] + '_mesh_depth.png'

    color_path = os.path.join(COLOR_ROOT, sequence_id, color_name)
    depth_path = os.path.join(DEPTH_ROOT, sequence_id, depth_name)
    render_depth_path = os.path.join(RENDER_DEPTH_ROOT, sequence_id, 'mesh_images', render_depth_name)

    entry = (color_path, depth_path, render_depth_path)

    train_file_paths.append(entry)

write_file_list('train_file_list.txt', train_file_paths)





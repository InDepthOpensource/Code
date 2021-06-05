import os, glob, random, re

# TEST
ROOT_DIRECTORY = r'/Users/user/Downloads/'
# Train
# ROOT_DIRECTORY = r'/usr/xtmp/user'

TEST_PERCENTAGE = 0.1

COLOR_ROOT = os.path.join(ROOT_DIRECTORY, 'matterport_undistorted_color/')
DEPTH_ROOT = os.path.join(ROOT_DIRECTORY, 'matterport_undistorted_depth/')
RENDER_DEPTH_ROOT = os.path.join(ROOT_DIRECTORY, 'matterport_render_depth/')


def write_file_list(filename, file_list):
    f = open(filename, 'w+')
    for i in file_list:
        print(' '.join(i), file=f)
    f.close()


if __name__ == '__main__':

    sequence_paths = [i[0] for i in os.walk(COLOR_ROOT)]

    # Each of the entry represent a experiments pair: (color.jpg, depth.png, render_depth.png)
    training_file_list = []
    test_file_list = []

    for sequence_path in sequence_paths:
        sequence_id = sequence_path.split(r'/')[-1]
        for color_name in os.listdir(sequence_path):
            if color_name.endswith('.jpg'):
                # Generate the rest of the filenames
                segments = re.split(r'[_.]', color_name)
                depth_name = segments[0] + '_' + segments[1].replace('i', 'd') + '_' + segments[2] + '.png'
                render_depth_name = segments[0] + '_' + segments[1].replace('i', 'd') + '_' + segments[
                    2] + '_mesh_depth.png'

                color_path = os.path.join(COLOR_ROOT, sequence_id, color_name)
                depth_path = os.path.join(DEPTH_ROOT, sequence_id, depth_name)
                render_depth_path = os.path.join(RENDER_DEPTH_ROOT, sequence_id, 'mesh_images', render_depth_name)

                entry = (color_path, depth_path, render_depth_path)

                # Decide if it's going to be train or test
                if random.random() < TEST_PERCENTAGE:
                    test_file_list.append(entry)
                else:
                    training_file_list.append(entry)

    write_file_list('train_file_list.txt', training_file_list)
    write_file_list('test_file_list.txt', test_file_list)

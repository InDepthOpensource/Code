import argparse
import os


def write_file_list(filename, file_list):
    f = open(filename, 'w+')
    for i in file_list:
        print(' '.join(i), file=f)
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file-list', type=str, default='')
    args = parser.parse_args()

    f = open(args.file_list, 'r')
    file_list = [i.split() for i in f.readlines()]
    f.close()

    # Normal looks like:   /usr/xtmp/user/matterport_render_normal/b8cTxDM8gDG/mesh_images/91b2405dc77f4ea7941586109ab53e7f_d1_2_mesh_ny.png
    # Rendered depth like: /usr/xtmp/user/matterport_render_depth/5ZKStnWn8Zo/mesh_images/4d00ab4dfa424fd3ad89792e92a39cfd_d1_1_mesh_depth.png
    for i in range(len(file_list)):
        rendered_depth_path = file_list[i][-1]
        nx_normal_path = rendered_depth_path.replace('matterport_render_depth', 'matterport_render_normal')
        nx_normal_path = nx_normal_path.replace('mesh_depth.png', 'mesh_nx.png')
        ny_normal_path = nx_normal_path.replace('mesh_nx.png', 'mesh_ny.png')
        nz_normal_path = nx_normal_path.replace('mesh_nx.png', 'mesh_nz.png')

        if os.path.exists(nx_normal_path) and os.path.exists(ny_normal_path) and os.path.exists(nz_normal_path):
            file_list[i] += [nx_normal_path, ny_normal_path, nz_normal_path]
        else:
            print(rendered_depth_path, 'corresponding normal is missing. Skip.')

    write_file_list('normal_list.txt', file_list)



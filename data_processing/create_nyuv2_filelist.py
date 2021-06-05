import os, re


def write_file_list(filename, file_list):
    f = open(filename, 'w+')
    for i in file_list:
        print(' '.join(i), file=f)
    f.close()


nyuv2_directories = ['/usr/xtmp/user/basements', '/usr/xtmp/user/bedrooms_part6', '/usr/xtmp/user/kitchens_part1',
                     '/usr/xtmp/user/office_kitchens', '/usr/xtmp/user/bathrooms_part1',
                     '/usr/xtmp/user/bedrooms_part7', '/usr/xtmp/user/kitchens_part2',
                     '/usr/xtmp/user/offices_part1', '/usr/xtmp/user/bathrooms_part2',
                     '/usr/xtmp/user/bookstore_part1', '/usr/xtmp/user/kitchens_part3',
                     '/usr/xtmp/user/offices_part2', '/usr/xtmp/user/bathrooms_part3',
                     '/usr/xtmp/user/bookstore_part2', '/usr/xtmp/user/libraries', '/usr/xtmp/user/playrooms',
                     '/usr/xtmp/user/bathrooms_part4', '/usr/xtmp/user/bookstore_part3',
                     '/usr/xtmp/user/living_rooms_part1', '/usr/xtmp/user/reception_rooms',
                     '/usr/xtmp/user/bedrooms_part1', '/usr/xtmp/user/cafe', '/usr/xtmp/user/living_rooms_part2',
                     '/usr/xtmp/user/scannet_render_depth', '/usr/xtmp/user/bedrooms_part2',
                     '/usr/xtmp/user/dining_rooms_part1', '/usr/xtmp/user/living_rooms_part3',
                     '/usr/xtmp/user/studies', '/usr/xtmp/user/bedrooms_part3', '/usr/xtmp/user/dining_rooms_part2',
                     '/usr/xtmp/user/living_rooms_part4', '/usr/xtmp/user/study_rooms',
                     '/usr/xtmp/user/bedrooms_part4', '/usr/xtmp/user/furniture_stores', '/usr/xtmp/user/misc_part1',
                     '/usr/xtmp/user/bedrooms_part5', '/usr/xtmp/user/home_offices', '/usr/xtmp/user/misc_part2']

# nyuv2_directories = ['/Users/user/Downloads/studies']


if __name__ == '__main__':
    # Each of the entry represent a experiments pair: (color.jpg, depth.png)
    training_file_list = []

    for base_directory in nyuv2_directories:
        sequence_paths = [i[0] for i in os.walk(base_directory)]
        for sequence_path in sequence_paths:
            sequence_id = sequence_path.split(r'/')[-1]
            for color_name in os.listdir(sequence_path):
                if color_name.endswith('.jpg'):
                    # Generate the rest of the filenames
                    segments = re.split(r'[.]', color_name)
                    depth_name = segments[0] + '.png'
                    color_path = os.path.join(base_directory, sequence_id, color_name)
                    depth_path = os.path.join(base_directory, sequence_id, depth_name)

                    if os.path.exists(depth_path):
                        entry = (color_path, depth_path)
                        training_file_list.append(entry)

    write_file_list('nyuv2_train_list.txt', training_file_list)

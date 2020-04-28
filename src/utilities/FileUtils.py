import os


def check_create_folder(folder_name):
    # print("seed is {}".format(seed))
    try:
        os.makedirs(folder_name)
    except OSError:
        if not os.path.isdir(folder_name):
            raise
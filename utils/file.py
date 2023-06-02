import os

class FileFilter(object):
    def __init__(self, accepted_extentions=['.png', '.jpg'], skip_basename=['imagesWithLables']):
        self.accepted_extensions = accepted_extentions
        self.skip_basename = skip_basename

    def __call__(self, filename, *args, **kwargs):
        base, ext = os.path.splitext(filename)
        base = os.path.basename(base)
        return ext in self.accepted_extensions and base not in self.skip_basename


def file_folder_exists_check(fph, folder_create=True):
    if not os.path.exists(fph):
        if folder_create:
            if os.path.splitext(fph)[-1] == '':
                os.makedirs(fph)
                # logger.info(f"create save folder: {fph}")
                # return true after create the folder.
                return True
            # if file is not exist then return false.
            return False
        else:
            # if folder is not exist then return false.
            return False
    return True

def file_check(file, ext=['.png', '.jpg'], skip_basename=[]):
    ft = FileFilter(ext, skip_basename)
    flag = ft(file)
    return flag

def file_generator1(data_path, ignore_folder_names=['AnnotationsVisualization', ], interest_file_exts=['.jpg', '.png',], max_depth=100):
    """

    :param data_path:
    :param ignore_folder_names:
    :param interest_file_exts:
    :return:
    """

    if not max_depth:
        return
    max_depth -= 1
    # logger.dubug(f"data_path")
    filenames = os.listdir(data_path)
    for file in filenames:
        filepath = os.path.join(data_path, file)
        # logger.debug(filepath)
        # begin to recursive
        if os.path.isdir(filepath):
            # skipping folders with name 'imagesWithLables'
            # if not file_check(filepath, ext=[''], skip_basename=['AnnotationsVisualization', 'Annotations']):
            if not file_check(filepath, ext=[''], skip_basename=ignore_folder_names):
                continue
            for file_path in file_generator1(filepath, ignore_folder_names=ignore_folder_names, interest_file_exts=interest_file_exts, max_depth=max_depth):
                yield file_path

        # begin to yield
        elif os.path.isfile(filepath):
            # filter files with suffix '.png'
            if not file_check(filepath, ext=interest_file_exts):
                continue
            yield filepath


def file_generator(data_path):

    data_path = os.path.join(data_path, 'JPEGImages')
    cnt = os.listdir(data_path)
    cnt = [os.path.join(data_path, ph) for ph in cnt]
    return cnt
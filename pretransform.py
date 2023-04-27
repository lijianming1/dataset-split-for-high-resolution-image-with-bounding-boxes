"""
author: lijianming
email: jmingl@tju.edu.cn
date: 2023.4.20
"""

import os
import re
import cv2
import logging
import numpy as np
import logging
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from enum import Enum
import xml.etree.ElementTree as ET
from utils import YamlParser, Color, NotImplementException, FileFileter
import shutil
CONF = YamlParser('conf.yaml')

levels = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'warn': logging.WARN,
    'error': logging.ERROR
}

LOGLEVEL = levels.get(CONF.get_by_url('global.log_level').lower())

logger = logging.getLogger(__name__)
console_handler = logging.StreamHandler()
console_handler.setLevel(LOGLEVEL)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

color_class = Color(6)
color_split = Color(20)

def show_images(imgs_list, imgs_titles=None, save_fig_name='all_image_crops.png'):
    """
    auto plot images in ceil(sqrt(len(image_list))) rows and columns
    use matplotlib object-oriented approach
    :param imgs_list:
    :param imgs_titles:
    :param save_fig_name:
    :return:
    """
    rc_c = int(np.ceil(np.sqrt(len(imgs_list))))
    rc_r = int(np.floor(len(imgs_list)/rc_c))
    fig, ax = plt.subplots(rc_r, rc_c)
    fig.set_facecolor('lightgrey')
    if not imgs_titles:
        imgs_titles = [f"crop{i}" for i in range(len(imgs_list))]
    for ind, (img, til) in enumerate(zip(imgs_list, imgs_titles)):
        coords = np.array([int(ind/rc_c), int(ind%rc_c)])
        # bgr to rgb
        try:
            ax[coords[0], coords[1]].imshow(img[:,:,::-1])
            ax[coords[0], coords[1]].set(title=til)
        except Exception as e:
            ax.imshow(img[:, :, ::-1])
            ax.set(title=til)
    if rc_c > 1:
        fig.suptitle('All Image Crops')

    plt.savefig(f'{save_fig_name}')
    # plt.show()

def file_folder_exists_check(fph, folder_create=True):
    if not os.path.exists(fph):
        if folder_create:
            if os.path.splitext(fph)[-1] == '':
                os.makedirs(fph)
                logger.info(f"create save folder: {fph}")
                # return true after create the folder.
                return True
            # if file is not exist then return false.
            return False
        else:
            # if folder is not exist then return false.
            return False
    return True

def file_check(file, ext=['.png', '.jpg'], skip_basename=[]):
    ft = FileFileter(ext, skip_basename)
    flag = ft(file)
    return flag


def read_image(img_path):
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    return img


def save_image(img, save_path):
    img_encode = cv2.imencode('.png', img)[1]
    img_encode.tofile(save_path)


def file_generator(data_path, ignore_folder_names=['AnnotationsVisualization', ], interest_file_exts=['.jpg', '.png',], max_depth=100):
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
            for file_path in file_generator(filepath, ignore_folder_names=ignore_folder_names, interest_file_exts=interest_file_exts, max_depth=max_depth):
                yield file_path

        # begin to yield
        elif os.path.isfile(filepath):
            # filter files with suffix '.png'
            if not file_check(filepath, ext=interest_file_exts):
                continue
            yield filepath



class TransFormType(Enum):
    BY_COUNT = 1
    BY_IMAGE_SIZE = 2
    @classmethod
    def get(cls, type):
        return cls.__getattr__(type.upper())


class Transform(object):
    def __init__(self, r, c, inter_size, split_type=TransFormType.BY_COUNT):
        """
        :param r: specified splitting row numbers or splitting crop's height size
        :param c:  specified splitting column numbers or splitting crop's width size
        :param inter_size: areas size of splitting intersection
        :param split_type: use (r, c) parameter as count number or crop size
        """
        # split row number
        self.r_num = r
        # split column number
        self.c_num = c
        # cross area size, must bigger than minimum size of defect
        self.inter_size = inter_size
        self.split_type = split_type


    def split_and_pad(self, image):
        """
        - split image into r rows and c columns;
        - pad (c_cum - w%c_num) at right most to get a divisible width, the same as height;
        - pad inter_size at left、ritht、top、bottom to get an image split in size: split_stride+2*inter_size
        :param image:
        :return:
        """
        h, w, c = image.shape
        # pad at the right most: pad righ t= (c_num - w%c_num) + inter_size
        # if w % self.c_num != 0:
        image = cv2.copyMakeBorder(image, top=0, bottom=0, left=0, right=int(self.c_num - w % self.c_num + self.inter_size), borderType=cv2.BORDER_CONSTANT, value=(114, 114, 114))

        # pad at the bottom most: pad bottom = (r_column - h%r_num) + inter_size
        # if h % self.r_num != 0:
        image = cv2.copyMakeBorder(image, top=0, bottom=int(self.r_num - h % self.r_num + self.inter_size), left=0, right=0, borderType=cv2.BORDER_CONSTANT, value=(114, 114, 114))

        # pad for all images at top and left most areas
        # pad top = inter_size， pad left = inter_size
        image = cv2.copyMakeBorder(image, top=int(self.inter_size), bottom=0, left=int(self.inter_size), right=0, borderType=cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return image

    def get_split_stride_rc_info(self, w, h):
        """
        compute split stride, row number and column number based on image width and hight
        :param w: image width
        :param h: image height
        :return: stride_w, stride_h, row_number, column_number
        """
        if self.split_type == TransFormType.BY_COUNT:
            stride_w = np.ceil(w / self.c_num)
            stride_h = np.ceil(h / self.r_num)
            r_num, c_num = self.r_num, self.c_num

        elif self.split_type == TransFormType.BY_IMAGE_SIZE:
            stride_w, stride_h = self.c_num, self.r_num
            r_num, c_num = int(np.ceil(h / self.r_num)), int(np.ceil(w / self.c_num))
        else:
            raise (NotImplementException('not implement transform split type'))

        logger.debug(f"split stride: stride_w={stride_w}, stride_h:={stride_h}")

        return stride_w, stride_h, r_num, c_num

    def get_all_split_blocks_coordinates(self, w, h):
        """
        compute all block coordinates base on image width and height
        :param w:
        :param h:
        :return: coordinates in list
        """
        stride_w, stride_h, r_num, c_num = self.get_split_stride_rc_info(w, h)
        blocks_coordinates = []
        for r in range(r_num):
            for c in range(c_num):
                logger.debug(f"row: {r}, column: {c}")
                x1, y1, x2, y2, = self.get_split_block_coord_after_pad(c, r, stride_w, stride_h, self.inter_size)
                logger.debug(f"block coordinates: ({x1},{y1},{x2},{y2})")
                blocks_coordinates.append([x1, y1, x2, y2])
        return blocks_coordinates

    def split(self, image, **kwargs):
        """
        - split image based on (row_num,column_num) parameters or specifying crop size
        - return stack split image arrays
        :param image: input image
        :param show: show image
        :return:  crops in numpy array
        """
        h, w, c = image.shape
        logger.debug(f'input image shape: w={w}, h={h}')

        # get a divisible images
        image = self.split_and_pad(image)
        logger.debug(f"input image after pad shape: {image.shape}")

        crops = []
        # split image by block bounding boxes coordinates
        image_crop = image.copy()

        coordinates = self.get_all_split_blocks_coordinates(w=w, h=h)
        for i, (x1, y1, x2, y2) in enumerate(coordinates):
            logger.debug(f"block coordinates: ({x1},{y1},{x2},{y2})")
            crop = image_crop[y1:y2, x1:x2, :]
            logger.debug(f"block size: {crop.shape}")
            crops.append(crop)

            if kwargs.get('show_image_with_crop_cells'):
                print(color_split(i))
                cv2.rectangle(image, (x1, y1), (x2,  y2), color=color_split(i), thickness=2)

        if kwargs.get('show_crops_all_in_one', True):
            show_images(imgs_list=crops, save_fig_name=kwargs.get('crops_all_in_one_name', 'all_crops_in_one.png'))
        if kwargs.get('show_image_with_crop_cells', True):
            show_images(imgs_list=[image], save_fig_name=kwargs.get('image_with_crops_cells_name', 'source_image_with_crops_rectangle.png'))

        return crops

    @staticmethod
    def get_split_block_coord_after_pad(c_index, r_index, stride_w, stride_h, inter):
        """
        get split block coordinates in a bounding box formats: x1,y1,x2,y2
        :return:
        """
        x1 = int(c_index*stride_w)
        y1 = int(r_index*stride_h)
        x2 = int(x1 + stride_w + 2*inter)
        y2 = int(y1 + stride_h + 2*inter)
        block_cord = [x1, y1, x2, y2]
        return block_cord

    def get_block_coord_by_index(self, block_index, img_w, img_h):
        """
        The acquisition of block coordinates relative to pad image based on block index
        :param block_index:
        :param img_w:
        :param img_h:
        :return:
        """
        # parse index to row number and column number.
        stride_w, stride_h, r_num, c_num = self.get_split_stride_rc_info(w=img_w, h=img_h)
        row_index = block_index // c_num
        col_index = block_index % c_num
        logger.debug(f"block index: {block_index} map index: ({row_index}, {col_index})")

        block_coord = self.get_split_block_coord_after_pad(c_index=col_index, r_index=row_index, stride_w=stride_w, stride_h=stride_h, inter=self.inter_size)
        logger.debug(f"the {block_index}th block coord: {block_coord}.")
        return block_coord

    def transform_bbox_coord_to_block_world(self, bbox_coord, assigned_block_index, img_w, img_h):
        """
        shift block coordinates to zero point
        :param bbox_coord: [n, 4]
        :param assigned_block_index: [n,]
        :param img_w:
        :param img_h:
        :return:
        """
        img_ws, img_hs = [img_w]*len(bbox_coord), [img_h]*len(bbox_coord)
        # shape: [n, 4]
        block_coords = np.array(list(map(self.get_block_coord_by_index, assigned_block_index, img_ws, img_hs)))

        # shift coordinates
        bbox_coord_block_world = bbox_coord - np.tile(block_coords[:,:2], (1, 2))

        return bbox_coord_block_world


    def bbox_block_assign(self, img_w, img_h, bboxes, bboxes_labels):
        """
        split bounding box into split blocks
        :param img_w:
        :param img_h:
        :param bboxes:
        :return:
        """
        def is_bbox_overlap_with_block(block, bbx):
            """
            compute the intersection between input bbox and block
            :param block:  coordinates
            :param bbx: bbx coordinates
            :return: intersection coordinates
            """
            # b_x1, b_y1, b_x2, b_y2 = bbx
            # bk_x1, bk_y1, bk_x2, bk_y2 = block

            max = np.maximum(bbx, block)
            min = np.minimum(bbx, block)
            intersection = np.concatenate([max[:2], min[2:]])
            return intersection

        # the returned coordinates are based on the padded image coordinates world.
        block_coordinates_in_pad_image = self.get_all_split_blocks_coordinates(w=img_w, h=img_h)
        assign_result = {}
        # iterate on all labeled bounding boxes
        for lb, bbx in zip(bboxes_labels, bboxes):
            for i, block_c in enumerate(block_coordinates_in_pad_image):
                # return the intersection box coordinates
                # intersection: [x1, y1, x2, y2]
                x1, y1, x2, y2 = is_bbox_overlap_with_block(block_c, bbx)
                if  x1 >= x2 and y1 >= y2:
                    # bbox doesn't overlap with block i
                    continue
                if not assign_result.get(i):
                    assign_result[i] = []
                assign_result[i].append([x1, y1, x2, y2, lb])
        return assign_result

    def bbox_block_assign_array(self, img_w, img_h, bboxes, bboxes_labels):
        """
        compute block indexes assigned with bounding boxes and corresponding labels
        :param img_w:
        :param img_h:
        :param bboxes:bounding boxes coordinates
        :param bboxes_labels: bounding boxes labels
        :return:
        """
        # shape: [r*c,4]
        block_coordinates = np.array(self.get_all_split_blocks_coordinates(w=img_w, h=img_h))
        # shape: [r*c, 4] -> [r*c, 1, 4]
        block_coordinates = block_coordinates[:,None]

        # adjust coordinates to the new pad image
        bboxes = bboxes+np.array([self.inter_size])
        # shape: [n, 4]
        bboxes = np.array(bboxes).reshape(-1, 4)
        # shape: [n, 4] -> [1, n, 4]
        bboxes = bboxes[None]

        # shape: [r*c, n, 2]
        top_left = np.maximum(block_coordinates[:,:,:2], bboxes[:,:,:2])
        # shape: [r*c, n, 2]
        bottom_right = np.minimum(block_coordinates[:,:,2:], bboxes[:,:,2:])
        # shape: [r*c, n, 4]
        bbox_assign = np.concatenate([top_left, bottom_right], axis=-1)
        # shape: [r*c, n]
        assign_flag = (bbox_assign[:,:,0] < bbox_assign[:,:,2]) & (bbox_assign[:,:,1] < bbox_assign[:,:,3])

        # shape: [n_true, 4]
        assigned_bbox_coordinates = bbox_assign[assign_flag]
        logger.debug(f"assigned bbox with coordinates: {assigned_bbox_coordinates}")

        # shape: [1, n]
        bboxe_labels = np.array(bboxes_labels).reshape(1, -1)
        # shape: [r*c, n]
        bboxe_labels = np.repeat(bboxe_labels, len(block_coordinates), axis=0)
        # shape: [n_true, ]
        assign_bbox_labels = bboxe_labels[assign_flag]
        logger.debug(f"assigned bbox labels: {assign_bbox_labels}")

        # shape: [n_ture, 2]
        block_index = np.argwhere(assign_flag)
        # shape: [n_true, 1]
        assign_block_index = block_index[:,0]
        logger.debug(f"assigned block indexes: {assign_block_index}")

        return assigned_bbox_coordinates, assign_bbox_labels, assign_block_index

    @staticmethod
    def make_info(filename, img_w, img_h, img_c, bboxes, bboxes_labels, blocks_indexes):
        infos = {}
        f = lambda d, **kwargs: d.update(**kwargs)
        sub = lambda d, ds, s: re.sub(d, ds, s)
        for ind, box, label in zip(blocks_indexes, bboxes, bboxes_labels):
            if infos.get(ind) is None:
                kwargs = {f'{ind}': {}}
                f(infos, **kwargs)
                f(
                    infos[str(ind)],
                    filename=sub('\.', f'_crop_{ind}.', filename),
                    size={},
                    objects=[]
                )

                f(
                    infos[str(ind)]['size'],
                    height=img_h,
                    width=img_w,
                    depth=img_c
                )
            temp = {}
            f(temp, name=label, bndbox={})
            f(temp['bndbox'], xmin=box[0], ymin=box[1], xmax=box[2], ymax=box[3])
            infos[str(ind)]['objects'].append(temp)

        return infos


class XMLParser(object):
    @classmethod
    def get_bbox_info(cls, xml_file):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        filename = root.find('filename').text
        width = int(root.find('size').find('width').text)
        height = int(root.find('size').find('height').text)
        objects = []
        for obj in root.findall('object'):
            label = obj.find('name').text
            bbox = [int(np.floor(eval(x.text))) for x in obj.find('bndbox')]
            objects.append({'label': label, 'bbox': bbox})

        xml_info = {}
        (lambda d, **kwargs: d.update(**kwargs)) (xml_info, filename=filename, size={'width':width, 'height':height}, objects=objects)

        return xml_info

    @classmethod
    def gen_xml_file(cls, bbox_info, xml_file_save='./test_out.xml'):
        """

        :param bbox_info: {
                                filename: xxx,
                                size: {
                                            height: xxx,
                                            width: xxx,
                                            depth: xxx
                                        }
                                objects:[
                                            {
                                                name: xxx
                                                bndbox:{
                                                    xmin: xxx,
                                                    ymin: xxx,
                                                    xmax: xxx,
                                                    ymax: xxx
                                                },

                                            },
                                           {
                                                name: xxx
                                                bndbox:{
                                                    xmin: xxx,
                                                    ymin: xxx,
                                                    xmax: xxx,
                                                    ymax: xxx
                                                },

                                            },,...
                                       ]
                            }
        :return:
        """

        def create_xml_tree(tree_depth=0, node=None, **kwargs):
            # for only one node
            if len(kwargs) == 1:
                # root node or sub node
                if tree_depth == 0:
                    # for root node
                    node = ET.Element(list(kwargs.keys())[0])
                    tree_depth += 1
                    # just jump in to next level
                    create_xml_tree(tree_depth, node=node, **dict(list(kwargs.values())[0]))
                    ET.dump(node)
                    tree = ET.ElementTree(node)
                    return tree
                else:
                    # for sub level with one node
                    if list(kwargs.keys())[0] == 'objects':
                        sub_node = node
                    else:
                        sub_node = ET.SubElement(node, list(kwargs.keys())[0])


                if not isinstance(list(kwargs.values())[0], str|int|list) and len(list(kwargs.values())[0]) > 1:
                    # for sub level has multiple nodes
                    tree_depth += 1
                    # jump into level with multiple nodes
                    create_xml_tree(tree_depth, sub_node, **dict(list(kwargs.values())[0]))
                else:
                    if not isinstance(list(kwargs.values())[0], list|dict):
                        sub_node.text = str(list(kwargs.values())[0])

                    else:
                        for obj in list(kwargs.values())[0]:
                            create_xml_tree(tree_depth, sub_node, object=obj)


            elif len(kwargs) > 1:
                # multiple nodes
                # jump into next tree layer
                tree_depth += 1
                for k, v in kwargs.items():
                    create_xml_tree(tree_depth, node, **{k:v})
                    ET.dump(node)


        tree = create_xml_tree(annotation=bbox_info)

        with open(xml_file_save, 'wb') as f:
            tree.write(f)

    @classmethod
    def xml_bbox_info_to_array(cls, xml_info):
        bbox_info = xml_info.get('objects')
        labels = []
        bboxes = []

        for box in bbox_info:
            labels.append(box.get('label'))
            xmin, ymin, xmax, ymax = box.get('bbox')
            bboxes.append([xmin, ymin, xmax, ymax])

        return np.array(labels), np.array(bboxes)



class TransformVOCDataset(object):
    def __init__(self, dataset_url, ignore_folder_names=['AnnotationsVisualization', 'Annotations'], interest_file_exts=['.jpg', '.png']):
        self.dataset_url = dataset_url
        self.ignore_folder_names = ignore_folder_names
        self.interest_file_exts = interest_file_exts
        self._class_name_list = None

    def gen_side_path_from_file_path(self, file_path):
        file_path_n = re.sub(self.dataset_url, self.dataset_url+'_trans', file_path)
        return file_path_n

    @property
    def sub(self):
        return lambda *args, f,: re.sub(args[0], args[1], f)

    def get_image_label_info_by_file_path(self, file_path):
        """you should modify this method if your dataset are not in VOC format"""
        # method for VOC dataset
        # sub = lambda *args, f=file_path,: re.sub(args[0], args[1], f)
        info_path = self.sub('JPEGImages','Annotations',f=self.sub('\..*$', '.xml', f=file_path))
        return info_path

    def copy_class_names_file(self):
        class_name_file_sour_path = next(file_generator(self.dataset_url,ignore_folder_names=[], interest_file_exts=['.txt'], max_depth=1))
        # if class name file doesn't exit then return False
        if class_name_file_sour_path is None:
            return None
        # acquisition of save path
        class_name_file_dest_path = self.gen_side_path_from_file_path(class_name_file_sour_path)
        # if already exist then return true.
        if file_folder_exists_check(class_name_file_dest_path):
            return class_name_file_dest_path
        # check folder exits before copy file
        file_folder_exists_check(os.path.dirname(class_name_file_dest_path), folder_create=True)
        # copy file
        shutil.copyfile(class_name_file_sour_path, class_name_file_dest_path)
        return class_name_file_dest_path

    @property
    def class_name_index(self):
        if self._class_name_list is None:
            file_path = self.copy_class_names_file()
            if file_path is None:
                raise NotImplementException('There is no class name file')
            with open(file_path, 'r') as f:
                classes = f.readlines()
                classes = [i.strip() for i in classes]
            self._class_name_list = classes

        return self._class_name_list


    def transform(self):
        R = CONF.get_by_url('Transform.r')
        C = CONF.get_by_url('Transform.c')
        INTER_SIZE = CONF.get_by_url('Transform.inter_size')
        SPLIT_TYPE = TransFormType.get(CONF.get_by_url('Transform.type'))
        SHOW_CROPS_WITH_ANNOTATIONS = CONF.get_by_url('global.show_crops_with_annotations')
        SHOW_CROPS_ALL_IN_ONE = CONF.get_by_url('global.show_crops_all_in_one')
        SHOW_IMAGE_WITH_CROP_CELLS = CONF.get_by_url('global.show_image_with_crop_cells')

        # copy class name file
        self.copy_class_names_file()
        # set up transform setup
        TF = Transform(r=R, c=C, inter_size=INTER_SIZE, split_type=SPLIT_TYPE)
        # get all image files and start to do transform task.
        image_generator = file_generator(self.dataset_url, ignore_folder_names=self.ignore_folder_names, interest_file_exts=self.interest_file_exts)

        for img_file in image_generator:
            img_file_side = self.gen_side_path_from_file_path(img_file)
            save_dir_name = os.path.dirname(img_file_side)
            file_folder_exists_check(save_dir_name, folder_create=True)
            save_file_name = os.path.basename(img_file_side)

            save_split_details_path = os.path.dirname(save_dir_name) + '/SplitDetails'
            file_folder_exists_check(save_split_details_path)
            crops_all_in_one_name = os.path.join(save_split_details_path, re.sub('\.jpg', '_all_in_one.jpg', save_file_name))
            image_with_crops_cells_name = os.path.join(save_split_details_path, re.sub('\.jpg', '_with_split_cells.jpg', save_file_name))

            logger.debug(f"save image path: {save_dir_name}, save image name: {save_file_name}")


            img = read_image(img_file)
            img_crops = TF.split(
                img,
                show_crops_all_in_one=SHOW_CROPS_ALL_IN_ONE,
                show_image_with_crop_cells=SHOW_IMAGE_WITH_CROP_CELLS,
                crops_all_in_one_name = crops_all_in_one_name,
                image_with_crops_cells_name=image_with_crops_cells_name
            )
            file_name, ext = os.path.splitext(save_file_name)

            for i, crop in enumerate(img_crops):

                save_img_file_path_name = os.path.join(save_dir_name, file_name+f'_crop_{i}'+ext)
                logger.debug(f"save crop image file path name: {save_img_file_path_name}")
                save_image(crop, save_img_file_path_name)



            # get iamge label_info_file path
            label_info_path = self.get_image_label_info_by_file_path(img_file)
            # true presents existing and vice verse
            if not file_folder_exists_check(label_info_path):
                # if label file is not exists then skip to next loop.
                continue
            # acquire label info
            img_info = XMLParser.get_bbox_info(label_info_path)
            # transform label info to array format
            bdbox_label_ar, bdbox_ar = XMLParser.xml_bbox_info_to_array(img_info)
            logger.debug(f"bdbox labels shape: {bdbox_label_ar.shape}")
            logger.debug(f"bdbox shape: {bdbox_ar.shape}")

            img_h, img_w, c = img.shape
            assigned_bbox_coordinates, assigned_bbox_labels, assigned_block_index = TF.bbox_block_assign_array(img_w, img_h, bboxes=bdbox_ar, bboxes_labels=bdbox_label_ar)
            logger.debug(f"bbox: {assigned_bbox_coordinates}")
            logger.debug(f"label:{assigned_bbox_labels}")
            logger.debug(f"assigned block index: {assigned_block_index}")


            # acquisition of new coordinates
            # [n, 4]
            assigned_bbox_coordinates = assigned_bbox_coordinates.reshape(-1, 4)
            # [n,]
            assigned_block_index = assigned_block_index.reshape(-1)
            # [n]
            assigned_bbox_labels = assigned_bbox_labels.reshape(-1)

            # transform original coordinates to block world.
            # [n, 4]
            assigned_bbox_coordinates_block_world = TF.transform_bbox_coord_to_block_world(assigned_bbox_coordinates, assigned_block_index, img_w, img_h)

            # save image info and bbox info to xml file
            crop_file_info = TF.make_info(filename=save_file_name, img_w=img_w, img_h=img_h, img_c=c, bboxes=assigned_bbox_coordinates_block_world.tolist(), bboxes_labels=assigned_bbox_labels.tolist(), blocks_indexes=assigned_block_index.tolist())

            logger.info(f"all crops xml file info: {crop_file_info}")

            # parser crops info and save it to xml file
            xml_infos = list(crop_file_info.values())

            # acquisition xml save path and save xml_info to xml file
            for xml_info in xml_infos:
                crop_file_name = xml_info.get('filename')
                save_crop_xml_file_path = self.gen_side_path_from_file_path(os.path.dirname(label_info_path))
                file_folder_exists_check(save_crop_xml_file_path, folder_create=True)

                crop_xml_file_name = re.sub('\.jpg$','.xml', crop_file_name )
                save_crop_xml_file_path = os.path.join(save_crop_xml_file_path, crop_xml_file_name)

                logger.debug(f"crop xml file save path: {save_crop_xml_file_path}")

                XMLParser.gen_xml_file(xml_info, save_crop_xml_file_path)

            if SHOW_CROPS_WITH_ANNOTATIONS:
                self.show_crop_with_annotation(self.gen_side_path_from_file_path(self.dataset_url)+'/Annotations')

    def show_crop_with_annotation(self, annotations_path):
        def parse_voc_xml_to_bbox(file_path):
            tree = ET.parse(file_path)
            root = tree.getroot()

            # get width and height of image
            width = int(root.find('size').find('width').text)
            height = int(root.find('size').find('height').text)

            bboxes = []
            labels = []
            for obj in root.findall('object'):
                label = obj.find('name').text
                xml_bbox = [int(np.floor(eval(x.text))) for x in obj.find('bndbox')]
                bboxes.append(xml_bbox)
                labels.append(label)
            return labels, bboxes


        annotations_gen = file_generator(data_path=annotations_path, ignore_folder_names=[], interest_file_exts=['.xml'])
        for xml_file_path in annotations_gen:
            crops_file_path = re.sub('Annotations', 'JPEGImages', xml_file_path)
            crops_file_path = re.sub('\.xml', '.jpg', crops_file_path)

            logger.debug(f"crops file: {crops_file_path}")
            logger.debug(f"xml file: {xml_file_path}")

            if not file_folder_exists_check(crops_file_path):
                logger.warn(f'crops file corresponding to annotation file not exist.')
                logger.warn(f"crops file path: {crops_file_path}")
                continue

            labels, bboxes = parse_voc_xml_to_bbox(xml_file_path)

            crop = read_image(crops_file_path)
            logger.debug(f"crop shape: {crop.shape}")
            for lb, bx in zip(labels, bboxes):
                cv2.rectangle(crop, bx[:2], bx[2:], color=color_class(self.class_name_index.index(lb)))

            # save crop image with annotations
            crop_with_annotations_path = re.sub('JPEGImages', 'AnnotationsVisualization', crops_file_path)
            file_folder_exists_check(os.path.dirname(crop_with_annotations_path), folder_create=True)
            save_image(crop, crop_with_annotations_path)





if __name__ == '__main__':
    pass
    #####################
    ## test Transform class function
    # # tf = Transform(408,408,10,TransFormType.BY_IMAGE_SIZE)
    # tf = Transform(2,3,10,TransFormType.BY_COUNT)
    # img_ph = '/home/ming/ming/project/object-detection/datasets/socks/imagesWithLables/微信图片_20230217153648_000_mcrhg.jpg'
    # img = cv2.imdecode(np.fromfile(img_ph, dtype=np.uint8), cv2.IMREAD_COLOR)
    # tf.split(img, True)

    ######################
    # # test XMLParser fucntion: get_bbox_info()
    # xml = XMLParser()
    # url = r'/windata/f/computer_vision/221213-袜子外观检测/100_for_test_resized_enhancement_VOC/Annotations/微信图片_20230217153636_000_mcrhg.xml'
    # res = xml.get_bbox_info(url)
    # print(res)
    # xml.gen_xml_file(res, 'recover.xml')

    #######################
    # # test XMLParser function: gen_xml_file()
    # info = {
    #     'filename': '微信图片_20230217153636_000_mcrhg.jpg',
    #     'size':{
    #         'width': 1000,
    #         'height': 600
    #     },
    #     'objects': [
    #         {
    #             'label': 'hole',
    #             'bndbox': {
    #                     'xmin': 731, 'ymin': 165, 'xmax': 778, 'ymax': 210
    #             }
    #         },
    #         {
    #              'label': 'dirt',
    #              'bbox': {
    #                  'xmin': 731, 'ymin': 165, 'xmax': 778, 'ymax': 210
    #              }
    #         },
    #     ]
    # }
    # xml = XMLParser()
    # xml.gen_xml_file(info)

    ######################
    # test for function: file_generator()
    # data_url = r'/windata/f/computer_vision/221213-袜子外观检测/100_for_test_resized_enhancement_VOC'
    # for ph in file_generator(data_url, interest_file_exts=['.xml'], max_depth=2):
    #     print(ph)

    #######################
    # data_url = r'/windata/f/computer_vision/221213-袜子外观检测/100_for_test_resized_enhancement_VOC'
    # file_url = r'/windata/f/computer_vision/221213-袜子外观检测/100_for_test_resized_enhancement_VOC/JPEGImages/微信图片_20230217153616_.jpg'
    # tfd = TransformVOCDataset(data_url)
    # res = tfd.get_image_label_info_by_file_path('/windata/f/computer_vision/221213-袜子外观检测/100_for_test_resized_enhancement_VOC/JPEGImages/微信图片_20230217153616.jpg')
    # print(res)

    ######################
    data_url = r'test/data_voc'
    tfd = TransformVOCDataset(data_url)
    tfd.transform()
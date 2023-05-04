"""
author: lijianming
email: jmingl@tju.edu.cn
date: 2023.4.20
"""

import os
import re
import cv2
import numpy as np
import logging
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

import xml.etree.ElementTree as ET
import shutil

from utils import FileFilter, file_folder_exists_check, file_check
from utils import read_image, save_image, show_images
from utils import file_generator, YamlParser
from utils import Color, NotImplementException
from utils import logger, CONF

from lib import TransFormType, Transform, XMLParser

color_class = Color(6)


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
    # data_url = r'test/data_voc'

    data_url = r'/windata/f/computer_vision/221213-袜子外观检测/100_for_test_resized_enhancement_VOC'

    tfd = TransformVOCDataset(data_url)
    tfd.transform()
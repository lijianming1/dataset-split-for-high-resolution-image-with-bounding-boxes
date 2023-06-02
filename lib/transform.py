import cv2
import re
import logging
from enum import Enum
from dataset_voc_split_with_bbox_for_high_resolution.utils import logger, Color, show_images
import numpy as np

color_split = Color(100)

class TransFormType(Enum):
    BY_COUNT = 1
    BY_IMAGE_SIZE = 2
    @classmethod
    def get(cls, type):
        if isinstance(type, str):
            return cls.__getattr__(type.upper())
        return cls.__getattr__(type)


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
        self.split_type = TransFormType.get(split_type) if isinstance(split_type, str) else split_type


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

        if len(image.shape) == 2:
            image = np.expand_dims(image, -1)

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
            raise (NotImplementedError('not implement transform split type'))

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
                # print(color_split(i))
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

    def transform_coord_from_block_world_to_original_image_world(self, bbox_coord, assigned_block_index, img_w, img_h):
        # shape: [1, -1] -> [n, 4]
        bbox_coord = np.array(bbox_coord).reshape(-1, 4)
        block_coord = np.array(self.get_block_coord_by_index(assigned_block_index, img_w, img_h)).reshape(-1, 4)

        # shift coordinates to original padded image
        # shape: [n, 4]
        bbox_coord = bbox_coord + np.tile(block_coord[:, :2], (1, 2))

        # shift coordinates to original image
        # shape: [n ,4]
        bbox_coord = bbox_coord - self.inter_size
        # if out of image size then truncate it to image size
        bbox_coord[bbox_coord < 0.0] = 0.0
        # x1, x2 less than image width
        bbox_coord[:, ::2] = np.where(bbox_coord[:, ::2] < img_w, bbox_coord[:, [0, 2]], img_w)
        # y1, y2 less than image height
        bbox_coord[:, 1::2] = np.where(bbox_coord[:, 1::2] < img_h, bbox_coord[:, 1::2], img_h)
        bbox_coord = bbox_coord.reshape(1, -1)
        return bbox_coord

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
            if infos.get(str(ind)) is None:
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



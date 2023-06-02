import cv2
from dataset_voc_split_with_bbox_for_high_resolution.utils import read_image
import numpy as np

class RotationShiftByCenter(object):
    """https://blog.csdn.net/MrR1ght/article/details/89196611"""
    @classmethod
    def gen_transform_mat(cls, degree, w, h):
        """rotation matrix plus shift matrix"""
        # change degree to radian
        radian = lambda x: x*(np.pi/180)

        # h, w = img.shape[:2]

        h_new = np.fabs(w*np.sin(radian(degree))) + np.fabs(h*np.cos(radian(degree)))
        w_new = np.fabs(w*np.cos(radian(degree))) + np.fabs(h*np.sin(radian(degree)))

        mat = cv2.getRotationMatrix2D((w/2, h/2), degree, 1)

        # add shift to rotation mat
        # shift in x direction
        mat[0, 2] += (w_new - w) / 2
        # shift in y direction
        mat[1, 2] += (h_new - h) / 2

        return mat



class TransformByMat(object):
    @classmethod
    def rotation_and_shift(cls, mat, img, w, h):
        img_n = cv2.warpAffine(img, mat, (w, h))
        return img_n

    @classmethod
    def transform_points_p2q(cls, mat: np.array, points: np.array) -> np.array:
        # mat: [2, 3]
        # points: [n, 2] -> [2, n]
        p_points = points.T
        # points: [2, n] -> [3, n]
        pad = np.ones((1, p_points.shape[1]))
        p_points = np.concatenate([p_points, pad], axis=0)

        # rotation and shift
        # [2, 3]@[3, n] -> [2, n]
        q_points = np.dot(mat, p_points).astype(np.int32)
        # [n, 2]
        q_points = q_points.T

        # # get top left point and bottom right point
        # shape: [n, 2] -> [n, 4]  [x1, y1, x2, y2]
        q_points = q_points.reshape(-1, 4)
        top_left = np.minimum(q_points[:, :2], q_points[:, 2:])
        bottom_right = np.maximum(q_points[:, :2], q_points[:, 2:])

        q_points = np.hstack([top_left, bottom_right]).reshape(-1, 2, 2)

        return q_points


def transform(img, p_points):
    """
    :param img: input image which need to be rotated by a degree
    :param p_points: input points need to map to transformed image
    :return: new image, q_points
    """
    h, w = img.shape[:2]
    p_points = np.array(p_points).astype(np.int32)
    # shape: [n, 2, 2] -> [n*2, 2]
    p_points = np.vstack(p_points)
    mat = RotationShiftByCenter.gen_transform_mat(90, w, h)

    img = TransformByMat.rotation_and_shift(mat, img, w=h, h=w)
    q_points = TransformByMat.transform_points_p2q(mat, p_points)

    return img, q_points


def transform_points(p_points, w, h, degree=90):
    p_points = np.array(p_points).astype(np.int32)
    p_points = np.vstack(p_points)
    mat = RotationShiftByCenter.gen_transform_mat(degree, w, h)
    q_points = TransformByMat.transform_points_p2q(mat, p_points)
    return q_points


def test_rotation_and_shift(img, rectangle_points):
    p_points = np.array(rectangle_points).astype(np.int32)
    for p in p_points:
        # draw rectangle on image
        cv2.rectangle(img, p[0,:].tolist(), p[1,:].tolist(), color=(0, 0, 255), thickness=2)
        """
        h, w = img.shape[:2]
        # get transform matrix
        mat = RotationShiftByCenter.gen_transform_mat(90, w, h)
        print(mat)
    
        # do transform based on matrix
        img_n = TransformByMat.rotation_and_shift(mat, img, w=h, h=w)
    
        # get q points by transform matrix
        q_points = TransformByMat.transform_points_p2q(mat, p_points)
        """
    img_n, q_points = transform(img, rectangle_points)
    q_points = q_points.reshape(-1, 2, 2)
    print(q_points)
    for q in q_points:
        # draw rectangle on the transformed image
        cv2.rectangle(img_n, q[0,:].tolist(), q[1,:].tolist(), color=(0, 255, 0), thickness=2)
    # cv2.imshow('res', img_n)
    # cv2.waitKey(0)
    # print(__file__)
    cv2.imwrite('../test/rotation_shift/test2.jpg', img_n)



if __name__ == '__main__':
    # url = 'test/rotation_shift/20230309135020.jpg'
    # url='/windata/f/computer_vision/221213-袜子外观检测/袜子原图-汇总/all_voc/JPEGImages/微信图片_20230225141831.jpg'
    url='/windata/f/computer_vision/221213-袜子外观检测/袜子原图-汇总/all_voc/JPEGImages/20230309135036.jpg'
    img = read_image(url)
    # labels = [[[581,1618],[696,1680]], [[906,1529],[929,1553]],[[523,1576],[544,1597]],[[479,1563],[498,1581]]]
    labels = [[[814, 1709], [835, 1735]], [[747, 1672],[772, 1696]]]
    # for lb in labels:
    #     rectangle = [lb]
    test_rotation_and_shift(img, rectangle_points=labels)

    '''
    [[[1709  445],[1735  466]]
     [[1672  508], [1696  533]]]
    '''










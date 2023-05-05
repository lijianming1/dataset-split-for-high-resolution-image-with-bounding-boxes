import cv2
from utils import read_image
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
        return q_points


def transform(img, p_points):
    """
    :param img: input image which need to be rotated by a degree
    :param p_points: input points need to map to transformed image
    :return: new image, q_points
    """
    h, w = img.shape[:2]
    p_points = np.array(p_points).astype(np.int32)
    mat = RotationShiftByCenter.gen_transform_mat(90, w, h)

    img = TransformByMat.rotation_and_shift(mat, img, w=h, h=w)
    q_points = TransformByMat.transform_points_p2q(mat, p_points)

    return img, q_points


def transform_points(p_points, w, h, degree=90):
    p_points = np.array(p_points).astype(np.int32)
    mat = RotationShiftByCenter.gen_transform_mat(degree, w, h)
    q_points = TransformByMat.transform_points_p2q(mat, p_points)
    return q_points


def test_rotation_and_shift(img, rectangle_points):
    p_points = np.array(rectangle_points).astype(np.int32)
    # draw rectangle on image
    cv2.rectangle(img, p_points[0].tolist(), p_points[1].tolist(), color=(0, 100, 255), thickness=2)
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
    img_n, q_points = transfrom(img, rectangle_points)
    # draw rectangle on the transformed image
    cv2.rectangle(img_n, q_points[0].tolist(), q_points[1].tolist(), color=(0, 255, 100), thickness=2)

    cv2.imwrite('test/rotation_shift/test.jpg', img_n)



if __name__ == '__main__':
    url = 'test/rotation_shift/20230309135020.jpg'
    img = read_image(url)
    x1, y1, x2, y2 = 327, 799, 372, 825
    rectangle = [(x1,y1), (x2, y2)]
    test_rotation_and_shift(img, rectangle_points=rectangle)










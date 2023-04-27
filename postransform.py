import cv2
class Transform(object):
    def __init__(self, r, c, inter_size):
        self.r_num = r
        self.c_num = c
        self.inter_size = inter_size

    def split_pad(self, image):
        w, h = image.shape
        if w % self.c_num != 0:
            image = cv2.copyMakeBorder(image, top=0, bottom=0, left=0, right=self.c_num - w % self.c_num, borderType=cv2.BORDER_CONSTANT, value=(114, 114, 114))
        if h % self.r_num != 0:
            image = cv2.copyMakeBorder(image, top=0, bottom=self.r_num - h % self.r_num, left=0, right=0, borderType=cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return image


    def split(self, image, inter_size=20):
        """
        - split image into r rows and n columns
        - return stack split image arrays
        :param image:
        :param inter_size:
        :return:
        """
        # pad image
        image = self.split_pad(image)
        w, h = image.shape
        stride_w = w // self.c_num
        stride_h = h // self.r_num
        split_lenth = stride_w + inter_size

        for c, r in zip(range(self.c_num-1), range(self.r_num-1)):

            self.get_zero_coord(c, r, )



        return


    def get_zero_coord(self, c_index, r_index, stride, inter):
        """

        :return:
        """
        x = min(c_index*stride - inter, 0)
        y = min(r_index*stride - inter, 0)
        return [x, y]



    def mask_region(self, ):
        pass

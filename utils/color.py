import numpy as np
from dataset_voc_split_with_bbox_for_high_resolution.utils.exception import NotImplementException

np.random.seed(20230520)
class Color(object):
    UNIFORM = True
    def __init__(self, color_num):
        self.color_num = color_num
        self._sliced_cmap = None

    @property
    def gen_color(self):
        if self._sliced_cmap is None:
            # cmap = plt.get_cmap('plasma')
            # sliced_cmap = cmap(np.linspace(0, 1, self.color_num))
            # self._sliced_cmap = sliced_cmap[:, :-1]
            self._sliced_cmap = np.random.randint(0, 256, size=(self.color_num, 3))

        return self._sliced_cmap


    def get_color(self, index):
        return self.gen_color[index]

    def __call__(self, index):
        if index >= self.color_num:
           raise NotImplementException('out of color map range: {0}'.format(index))

        color = self.get_color(index)
        if self.UNIFORM:
            color = color.astype(np.int32)
            # color = (color*256).astype(np.int32)
        return color.tolist()


if __name__ == '__main__':
    c = Color(3)
    print(c.gen_color)
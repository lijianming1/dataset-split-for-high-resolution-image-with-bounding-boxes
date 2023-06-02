import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def read_image(img_path):
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    return img


def save_image(img, save_path):
    img_encode = cv2.imencode('.png', img)[1]
    img_encode.tofile(save_path)


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
    rc_r = int(np.ceil(len(imgs_list)/rc_c))
    fig, ax = plt.subplots(rc_r, rc_c)
    fig.set_facecolor('lightgrey')
    if not imgs_titles:
        imgs_titles = ["crop{0}".format(i) for i in range(len(imgs_list))]
    for ind, (img, til) in enumerate(zip(imgs_list, imgs_titles)):
        coords = np.array([int(ind/rc_c), int(ind%rc_c)])
        # bgr to rgb
        try:
            if len(ax.shape) < 2:
                ax[coords[1]].imshow(img[:,:,::-1])
                ax[coords[1]].set(title=til)
            else:
                ax[coords[0], coords[1]].imshow(img[:,:,::-1])
                ax[coords[0], coords[1]].set(title=til)
        except Exception as e:
            ax.imshow(img[:, :, ::-1])
            ax.set(title=til)
    if rc_c > 1:
        fig.suptitle('All Image Crops')

    plt.savefig(save_fig_name)
    # plt.show()
    plt.close()


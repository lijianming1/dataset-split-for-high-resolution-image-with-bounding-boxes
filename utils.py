from ruamel import yaml
import time
import os
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

class YamlParser(object):
    def __init__(self, config_path):
        self.config_path = config_path
        self._config = None
        self.latest_time = None

    def is_obsolete(self, life_time=3):
        if not self.latest_time:
            return False
        delta = time.time() - self.latest_time
        if life_time < delta:
            return True
        return False

    def _load_config(self, load_path=None):
        if not load_path:
            load_path = self.config_path
        with open(load_path, 'r') as f:
            self._config = yaml.load(f, Loader=yaml.Loader)
        self.latest_time = time.time()

    @property
    def config(self):
        if self._config is None:
            self._load_config()
        return self._config

    def get(self, key):
        if not self.config or self.is_obsolete(life_time=3):
            self._load_config()
        return self.config.get(key)

    def get_by_url(self, url):
        def get_recursive(d, k):
            ks = k.split('.', 1)
            if len(ks) > 1:
                return get_recursive(d.get(ks[0]), ks[-1])

            return d.get(k)

        v = get_recursive(self.config, url)
        return v

    def get_sections(self):
        if not self.config:
            self._load_config()

        return list(self.config.keys())


class GlobalConfigFile(YamlParser):
    def __init__(self, config_path):
        super().__init__(config_path)

    def set(self, key, value):
        # keys = key.split('.', 1)
        if not self.config:
            self._load_config()

        return self.recursive_set(self.config, key, value)

    @staticmethod
    def recursive_set(d, k, v):
        keys = k.split('.', 1)
        if len(keys) > 1:
            return GlobalConfigFile.recursive_set(d.get(keys[0]), keys[-1], v)
        d.update({keys[0]: v})
        return 0

    def dump(self, dump_path=None):
        if not dump_path:
            dump_path = self.config_path

        # backup already existing configure file.
        if os.path.exists(dump_path):
            bk_name = os.path.splitext(dump_path)
            bk_name = bk_name[0] + "_backup" + bk_name[-1]
            os.replace(dump_path, bk_name)

        with open(dump_path, 'x') as f:
            if not self.config:
                self._load_config()

            yaml.dump(self.config, f, Dumper=yaml.RoundTripDumper)



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
           raise NotImplementException(f'out of color map range: {index}')

        color = self.get_color(index)
        if self.UNIFORM:
            color = color.astype(np.int32)
            # color = (color*256).astype(np.int32)
        return color.tolist()

class NotImplementException(Exception):
    def __init__(self, error_info):
        self.error_info = error_info
        super().__init__(self)

    def __str__(self):
        return self.error_info

class FileFileter(object):
    def __init__(self, accepted_extentions=['.png', '.jpg'], skip_basename=['imagesWithLables']):
        self.accepted_extensions = accepted_extentions
        self.skip_basename = skip_basename

    def __call__(self, filename, *args, **kwargs):
        base, ext = os.path.splitext(filename)
        base = os.path.basename(base)
        return ext in self.accepted_extensions and base not in self.skip_basename


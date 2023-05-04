from ruamel import yaml
import time
import os


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



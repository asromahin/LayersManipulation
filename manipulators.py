from utils import only_chars
from layers import MLayer


class LayersManipulation:
    def __init__(self, model):
        self.model = model
        self.read_model()

    def refresh(self):
        self.read_model()

    def read_model(self):
        self.layers = self._read_model(self.model)
        self.groups = self._group_layers(self.layers)
        self.groups_keys = list(self.groups.keys())

    def _read_model(self, m):
        all_layers = []
        for key in dir(m):
            value = getattr(m, key)
            str_key = str(type(value))
            if not 'method' in str_key:
                if '_modules' in dir(value):
                    layer = MLayer(key, m, value)
                    all_layers.append(layer)
        for n, ch in m.named_children():
            all_layers += self._read_model(ch)
        return all_layers

    def _group_layers(self, layers):
        res_dict = {}
        for layer in layers:
            str_key = only_chars(layer.key)
            data = res_dict.get(str_key)
            if data is None:
                res_dict[str_key] = [layer]
            else:
                res_dict[str_key].append(layer)
        return res_dict

    def _is_group_exists(self, group_name):
        if group_name in self.groups_keys:
            return True
        else:
            raise Exception(
                f'Group with name "{group_name}" does not exists.\nYou can use one of this list:\n{self.groups_keys}'
            )
            return False

    def change_group(self, group_name, new_module):
        if self._is_group_exists(group_name):
            for layer in self.groups[group_name]:
                n_layer = new_module(layer.layer)
                layer.change_module(n_layer)

    def change_group_layer(self, group_name, new_module, index):
        if self._is_group_exists(group_name):
            layer = self.groups[group_name][index]
            n_layer = new_module(layer.layer)
            layer.change_module(n_layer)
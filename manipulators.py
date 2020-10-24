from layers import MLayer, MLayerList


class LayersManipulation:
    def __init__(self, model):
        self.model = model
        self.read_model()

    def refresh(self):
        self.read_model()

    def read_model(self):
        self.layers = self._read_model(self.model)
        self.groups = self.layers.group_layers()
        self.groups_keys = list(self.groups.keys())

    def _read_model(self, m, parent_path=''):
        all_layers = MLayerList()
        path = ''
        for key in dir(m):
            value = getattr(m, key)
            str_key = str(type(value))
            if not 'method' in str_key:
                if '_modules' in dir(value):
                    path = '/'.join([parent_path, key])
                    layer = MLayer(key, m, value, path)
                    all_layers.append(layer)
        for n, ch in m.named_children():
            all_layers += self._read_model(ch, path)
        return all_layers

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
        self.refresh()

    def change_group_layer(self, group_name, new_module, index):
        if self._is_group_exists(group_name):
            layer = self.groups[group_name][index]
            n_layer = new_module(layer.layer)
            layer.change_module(n_layer)
        self.refresh()

    def apply_for_layers(self, new_module, layers=None):
        if layers is None:
            layers = self.layers
        for layer in layers:
            n_layer = new_module(layer.layer)
            layer.change_module(n_layer)
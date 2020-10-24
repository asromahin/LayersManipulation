import torch
from utils import only_chars


class MLayer:
    def __init__(self, key, parent_layer, layer, path=''):
        self.key = key
        self.parent_layer = parent_layer
        self.layer = layer
        self.is_list = type(self.layer) == torch.nn.ModuleList
        self.path = path

    def get_children(self, layer=None, recursive=False):
        if layer is None:
            layer = self.layer
        children = []
        for n, ch in layer.named_children():
            children.append(ch)
            if recursive:
                children += self.get_children(layer, True)
        return children

    def __repr__(self):
        return str(type(self))+'_'.join([self.key])

    def change_module(self, new_module):
        if not self.is_list:
            setattr(self.parent_layer, self.key, new_module)
        else:
            raise Exception('You try change the list!')


class MLayerList(list):
    def __init__(self):
        super(MLayerList, self).__init__()

    def append(self, layer: MLayer) -> None:
        super(MLayerList, self).append(layer)

    def group_layers(self):
        res_dict = {}
        for layer in self:
            #str_key = only_chars(layer.key)
            str_key = only_chars(str(type(layer.layer)))
            data = res_dict.get(str_key)
            if data is None:
                res_dict[str_key] = [layer]
            else:
                res_dict[str_key].append(layer)
        return res_dict



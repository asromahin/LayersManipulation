import torch


class MLayer:
    def __init__(self, key, parent_layer, layer, path=''):
        self.key = key
        self.parent_layer = parent_layer
        self.layer = layer
        self.is_list = type(self.layer) == torch.nn.ModuleList
        self.path = path

    def get_children(self):
        children = []
        for n, ch in self.layer.named_children():
            children.append(ch)
        return children

    def __repr__(self):
        return str(type(self))+'_'.join([self.key])

    def change_module(self, new_module):
        if not self.is_list:
            setattr(self.parent_layer, self.key, new_module)
        else:
            raise Exception('You try change the list!')

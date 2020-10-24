import torch


def get_add(set_block, call_after=True):
    class AddLayer(torch.nn.Module):
        def __init__(
                self,
                old_block,
                new_block=set_block,
        ):
            super(AddLayer, self).__init__()
            self.old_block = old_block
            self.new_block = new_block

        def forward(self, *args):
            if call_after is False:
                args = self.new_block(args)
            args = self.old_block(*args)
            if call_after:
                args = self.new_block(*args)
            return args
    return AddLayer


def get_add(set_block, call_after=True, set_index=-1, after_set=True):
    class AddLayerInList(torch.nn.Module):
        def __init__(
                self,
                old_list,
                new_block=set_block,
                index=set_index,
        ):
            super(AddLayerInList, self).__init__()
            self.old_list = old_list
            self.new_block = new_block
            self.index = index
            self.new_list = self.create_new_list(self.old_list, self.new_block, index)

        def create_new_list(self, old_list, set_block, index):
            new_list = torch.nn.ModuleList()
            for i, layer in enumerate(old_list):
                if i == index and not after_set:
                    new_list.append(set_block)
                new_list.append(layer)
                if i == index and after_set:
                    new_list.append(set_block)
            return new_list

        def forward(self, *args):
            if call_after is False:
                args = self.new_block(args)
            args = self.old_block(*args)
            if call_after:
                args = self.new_block(*args)
            return args

    return AddLayerInList

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

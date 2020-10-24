import torch


def get_transform(transform_function):
    class TransformLayer(torch.nn.Module):
        def __init__(
                self,
                old_block,
                transform=transform_function,
        ):
            super(TransformLayer, self).__init__()
            self.old_block = old_block
            self.transform = transform

        def forward(self, *args):
            self.transform(self.old_block)
            args = self.old_block(*args)
            return args
    return TransformLayer


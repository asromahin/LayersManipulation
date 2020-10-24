import torch


def callback_pass(x):
    return x


def get_callback_layer(callback_function=callback_pass):
    class CallbackLayer(torch.nn.Module):
        def __init__(
                self,
                old_block,
                callback=callback_function,
        ):
            super(CallbackLayer, self).__init__()
            self.old_block = old_block
            self.callback = callback

        def forward(self, *args):
            x = self.old_block(*args)
            x = self.callback(x)
            return x
    return CallbackLayer





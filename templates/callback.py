import torch


def callback_pass(x):
    return x


def get_callback(callback_function=callback_pass, call_after=True):
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
            if call_after is False:
                args = self.callback(args)
            x = self.old_block(*args)
            if call_after:
                x = self.callback(x)
            return x
    return CallbackLayer





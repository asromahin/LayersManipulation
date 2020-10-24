import torch


def callback_pass(x):
    return x


class CallbackLayer(torch.nn.Module):
    def __init__(
            self,
            old_encoder_block,
            callback=callback_pass,
    ):
        super(CallbackLayer, self).__init__()
        self.old_block = old_encoder_block
        self.callback = callback

    def forward(self, *args):
        x = self.old_block(*args)
        x = self.callback(x)
        return x





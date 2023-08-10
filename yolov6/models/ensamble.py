import torch
import torch.nn as nn


class EnsambleModel(nn.Module):
    def __init__(self, model_player, model_phone):
        super().__init__()
        self.model_player = model_player
        self.model_phone = model_phone
        assert model_player.stride.equal(model_phone.stride)
        self.stride = model_player.stride

    def forward(self, im):
        y_player = self.model_player(im)
        f_player = None
        if isinstance(y_player, list):
            y_player, f_player = y_player[0], y_player[1]
        y_phone = self.model_phone(im)
        f_phone = None
        if isinstance(y_phone, list):
            y_phone, f_phone = y_phone[0], y_phone[1]
        y = tuple(torch.cat([y_player_s, y_phone_s], 1)
                  for y_player_s, y_phone_s in zip(y_player, y_phone))
        return y, f_player

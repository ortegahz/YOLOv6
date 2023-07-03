import torch
import torch.nn as nn


class EnsambleModel(nn.Module):
    def __init__(self, model_player, model_phone):
        super().__init__()
        self.model_player = model_player
        self.model_phone = model_phone

    def forward(self, im):
        y_player = self.model_player(im)
        y_player = y_player[0] if isinstance(y_player, list) else y_player
        y_phone = self.model_phone(im)
        y_phone = y_phone[0] if isinstance(y_phone, list) else y_phone
        y = tuple(torch.cat([y_player_s, y_phone_s], 1)
                  for y_player_s, y_phone_s in zip(y_player, y_phone))
        return y

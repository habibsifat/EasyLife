from typing import TypeVar

import torch
import torch.nn as nn

from GameOfLifeBase import GameOfLifeBase


T = TypeVar('T', bound='GOLF_1')
class GameOfLifeForward_1(GameOfLifeBase):
    def __init__(self):
        super().__init__()

        self.layers = nn.ModuleList([
            nn.Conv2d(in_channels=1,    out_channels=1,    kernel_size=(3,3), padding=1, padding_mode='circular'),
            nn.Conv2d(in_channels=1+1,  out_channels=2,    kernel_size=(1,1)),
            nn.Conv2d(in_channels=2,    out_channels=1,    kernel_size=(1,1)),
        ])
        self.dropout    = nn.Dropout(p=0.0)
        self.activation = nn.PReLU()


    def forward(self, x):
        x = input = self.cast_inputs(x)
        for n, layer in enumerate(self.layers):
            if n == 1:                               # autodetect 1+in_channels
                x = torch.cat([ x, input ], dim=1)   # passthrough original cell state
            x = layer(x)
            if n != len(self.layers)-1:
                x = self.activation(x)
                x = self.dropout(x)
            else:
                x = torch.sigmoid(x)  # output requires sigmoid activation
        return x


    def weights_init(self, layer):
        ### Default initialization seems to work best, at least for Z shaped ReLU1 - see GameOfLifeHardcodedReLU1_21.py
        if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
            ### kaiming_normal_ corrects for mean and std of the relu function
            if isinstance(self.activation, (nn.ReLU, nn.LeakyReLU, nn.PReLU)):
                nn.init.kaiming_normal_(layer.weight)
                # nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    # small positive bias so that all nodes are initialized
                    nn.init.constant_(layer.bias, 0.1)
        else:
            # Use default initialization
            pass



if __name__ == '__main__':
    from train import train
    import numpy as np

    model = GameOfLifeForward_1().load(load_weights=False)
    model.print_params()
    print('-' * 20)


    board = np.array([
        [0,0,0,0,0],
        [0,0,0,0,0],
        [0,1,1,1,0],
        [0,0,0,0,0],
        [0,0,0,0,0],
    ])
    result1 = model.predict(board)
    result2 = model.predict(result1)

    train(model, batch_size=100, grid_size=5, accuracy_count=100_000)

    result3 = model.predict(board)
    result4 = model.predict(result3)
    assert np.array_equal(board, result4)

    print('-' * 20)
    model.print_params()
import torch


class Bottlrneck(torch.nn.Module):
    def __init__(self, In_channel, Med_channel, Out_channel, downsample=False):
        super(Bottlrneck, self).__init__()
        self.stride = 1
        if downsample == True:
            self.stride = 2

        self.layer = torch.nn.Sequential(
            torch.nn.Conv1d(In_channel, Med_channel, 1, self.stride),
            torch.nn.BatchNorm1d(Med_channel),
            torch.nn.ReLU(),
            torch.nn.Conv1d(Med_channel, Med_channel, 3, padding=1),
            torch.nn.BatchNorm1d(Med_channel),
            torch.nn.ReLU(),
            torch.nn.Conv1d(Med_channel, Out_channel, 1),
            torch.nn.BatchNorm1d(Out_channel),
            torch.nn.ReLU(),
        )

        if In_channel != Out_channel:
            self.res_layer = torch.nn.Conv1d(In_channel, Out_channel, 1, self.stride)
        else:
            self.res_layer = None

    def forward(self, x):
        if self.res_layer is not None:
            residual = self.res_layer(x)
        else:
            residual = x
        return self.layer(x) + residual


class ResNet(torch.nn.Module):
    def __init__(self, in_channels=2, classes=5):
        super(ResNet, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels, 8, kernel_size=7, stride=2, padding=3),
            torch.nn.MaxPool1d(3, 2, 1),

            Bottlrneck(8, 8, 32, False),
            Bottlrneck(32, 8, 32, False),
            Bottlrneck(32, 8, 32, False),
            #
            Bottlrneck(32, 16, 64, True),
            Bottlrneck(64, 16, 64, False),
            Bottlrneck(64, 16, 64, False),
            #
            Bottlrneck(64, 32, 128, True),
            Bottlrneck(128, 32, 128, False),
            Bottlrneck(128, 32, 128, False),
            #
            Bottlrneck(128, 64, 256, True),
            Bottlrneck(256, 64, 256, False),
            Bottlrneck(256, 64, 256, False),

            torch.nn.AdaptiveAvgPool1d(1)
        )
        self.classifer = torch.nn.Sequential(
            torch.nn.Linear(256, classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 256)
        x = self.classifer(x)
        return x


if __name__ == '__main__':
    x = torch.randn(size=(32, 55, 16))
    model = ResNet(in_channels=55, classes=38)

    output = model(x)
    print(model)

import torchvision
import torch.nn as nn

class DenseNet121(nn.Module):
    """
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x


class Vgg16(nn.Module):
    def __init__(self, out_size):
        super(Vgg16, self).__init__()
        self.vgg16 = torchvision.models.vgg16(pretrained=True)
        num_ftrs = self.vgg16.classifier[0].in_features
        self.vgg16.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.vgg16(x)
        return x


class Vgg19(nn.Module):
    def __init__(self, out_size):
        super(Vgg19, self).__init__()
        self.vgg19 = torchvision.models.vgg19(pretrained=True)
        num_ftrs = self.vgg19.classifier[0].in_features
        self.vgg19.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.vgg19(x)
        return x


def getmodel(modelname, nn):
    switcher = {
        "DenseNet121":  DenseNet121(nn).cuda(),
        "Vgg16": Vgg16(nn).cuda(),
        "Vgg19": Vgg19(nn).cuda()
    }

    model = switcher.get(modelname, lambda: "Invalid Model")
    return model

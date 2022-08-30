import torch
import torch.nn.functional as F
import torchvision


class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super().__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for block in blocks:
            for param in block.parameters():
                param.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = F.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, *args, **kwargs):
        pred, target = args[0], args[1]
        feature_layers = kwargs.get("feature_layers", [0, 1, 2])
        feature_weights = kwargs.get("feature_weights", [0.65, 0.3, 0.05])
        style_layers = kwargs.get("style_layers", [])

        if pred.shape[1] != 3:
            pred = pred.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        pred = (pred-self.mean) / self.std
        target = (target-self.mean) / self.std
        if self.resize:
            pred = self.transform(pred, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = pred
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += feature_weights[i]*F.mse_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += F.mse_loss(gram_x, gram_y)
        return loss

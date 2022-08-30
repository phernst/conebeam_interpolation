import torch
import torch.nn.functional as F
from torchgeometry.losses import SSIM


class RecoLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

        sigma = (1.5,)
        num_scale = len(sigma)
        width = 11

        self.weights = torch.zeros(num_scale, width, width, device='cuda')

        for i in range(num_scale):
            gaussian = (-1.*torch.arange(-(width/2), width/2)**2/(2*sigma[i]**2)).exp()
            gaussian = torch.outer(gaussian, gaussian)
            gaussian = gaussian/gaussian.sum()
            self.weights[i] = gaussian

        self.msssim = SSIM(
            window_size=width,
            reduction='mean',
            # sigma=sigma,
            # data_range=1.,
            # reduction='none',
        )

        self.alpha = 0.025

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction = prediction.to(target.dtype)
        absdiff = (prediction - target).abs()

        convweights = self.weights[-1:, None].tile(target.shape[1], target.shape[1], 1, 1)
        w_absdiff = F.conv2d(absdiff, weight=convweights, stride=1, padding=self.weights.shape[-1]//2)

        ssim_pred = prediction.clamp(min=0, max=1)
        ssim_target = target.clamp(min=0, max=1)
        l_ssim = 1 - self.msssim(ssim_pred, ssim_target)
        l_l1 = w_absdiff.mean()

        return self.alpha*l_ssim + (1 - self.alpha)*l_l1

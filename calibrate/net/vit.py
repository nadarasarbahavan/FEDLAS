import torch
import torch.nn as nn
import timm

class vit_base_patch16(nn.Module):
    def __init__(self, model_name='vit_base_patch16_384',
                 num_classes=100,
                 pretrained=True,
                 **kwargs):
        super().__init__()

        self.vit = timm.create_model(
            model_name,
            num_classes=num_classes,
            pretrained=pretrained
        )
        # classifier head
        self.fc = self.vit.head

    def forward(self, x, return_features=False):
        x = self.vit.forward_features(x)

        cls_token = x[:, 0]                # (B, D)
        preds = self.fc(cls_token)

        if return_features:
            return preds,cls_token.detach()
        else:
            return preds




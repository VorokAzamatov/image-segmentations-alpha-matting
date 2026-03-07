from torch import nn
from segmentation_models_pytorch.utils.losses import DiceLoss

class BCEDiceLoss(nn.Module):
    """
    Combined loss: BCEWithLogits + DiceLoss.
    You can set weights for each component.
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5, dice_activation='sigmoid'):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(activation=dice_activation)

    def forward(self, y_pred, y_true):
        bce_loss = self.bce(y_pred, y_true)
        dice_loss = self.dice(y_pred, y_true)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss
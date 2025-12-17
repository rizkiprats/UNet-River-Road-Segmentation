import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

def get_loss():
    bce = nn.BCEWithLogitsLoss()
    dice = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

    def loss_fn(pred, target):
        bce_loss = bce(pred, target)
        dice_loss = dice(pred, target)
        return 0.7 * bce_loss + 0.3 * dice_loss  # bisa ubah weight sesuai kebutuhan

    return loss_fn

def get_loss_v2():
    TverskyLoss = smp.losses.TverskyLoss(mode = 'multilabel', log_loss = False)
    BCELoss     = smp.losses.SoftBCEWithLogitsLoss()
    
    def loss_fn(pred, target):
        return 0.7 * BCELoss(pred, target) + 0.3 * TverskyLoss(pred, target)
    
    return loss_fn

def get_loss_v2_multiclass():
    ce = torch.nn.CrossEntropyLoss()
    tversky = smp.losses.TverskyLoss(mode='multiclass', log_loss=False)

    def loss_fn(pred, target):
        # target harus long untuk cross-entropy
        if target.dtype != torch.long:
            target = target.long()
        return 0.5 * ce(pred, target) + 0.5 * tversky(pred, target)
    
    return loss_fn

def get_loss_v2_multilabel():
    tversky = smp.losses.TverskyLoss(mode='multilabel', log_loss=False)
    bce     = smp.losses.SoftBCEWithLogitsLoss()

    def loss_fn(pred, target):
        # pred, target: [B, C, H, W]
        return 0.5 * bce(pred, target) + 0.5 * tversky(pred, target)
    return loss_fn
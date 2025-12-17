import segmentation_models_pytorch as smp

def get_model(num_classes=1):
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes,
        activation=None
    )
    
    # return smp.Unet(
    #     encoder_name="mit_b2",        # atau "efficientnet-b4", "se_resnext50_32x4d"
    #     encoder_weights="imagenet",
    #     in_channels=3,
    #     classes=num_classes,  # atau jumlah kelas kamu
    #     activation=None
    # )
    
def get_model_upernet(num_classes=1):
    return smp.UPerNet(
        encoder_name='resnet34',
        encoder_depth=5,
        encoder_weights='imagenet',
        decoder_channels=256,
        decoder_use_norm='batchnorm',
        in_channels=3,
        classes=num_classes,
        activation=None,
        upsampling=4,
        aux_params=None
    )

def get_model_unet_resnet101(num_classes=1):
    return smp.Unet(
        encoder_name='resnet101',
        encoder_weights='imagenet',
        in_channels=3,
        classes=num_classes,
        activation=None,
    )
import segmentation_models_pytorch as smp

def get_model(num_classes=1):
    if num_classes <= 0:
        raise ValueError("num_classes must be greater than zero.")
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes,
        activation=None
    )
    
def get_model_upernet(num_classes=1):
    if num_classes <= 0:
        raise ValueError("num_classes must be greater than zero.")
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
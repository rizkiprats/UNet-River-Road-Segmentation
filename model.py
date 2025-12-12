import segmentation_models_pytorch as smp

def get_model():
    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None
    )
    
def get_model_upernet():
    return smp.UPerNet(
        encoder_name='resnet34', 
        encoder_depth=5, 
        encoder_weights='imagenet', 
        decoder_channels=256, 
        decoder_use_norm='batchnorm', 
        in_channels=3, 
        classes=1, 
        activation=None, 
        upsampling=4, 
        aux_params=None
    )
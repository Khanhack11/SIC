"""Candidate field classifier contract, separate from the deployed V2 checkpoint."""
import io
from PIL import Image, ImageOps
from torch import nn
from torchvision import models, transforms
from backend.config import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD
from backend.preprocessing import orient_rgb

FIELD_CLASSES = CLASS_NAMES + ['Unsupported']
FIELD_ARCHITECTURE = 'mobilenet_v3_large_field_v1'
FIELD_PREPROCESSING = 'letterbox_224_rgb_jpegdraft1024_imagenet_v1'


def decode_field_image(raw):
    with Image.open(io.BytesIO(raw)) as image:
        # JPEG decoder downsampling avoids full-resolution copies before the 224px model input.
        image.draft('RGB',(1024,1024))
        return orient_rgb(image)


def build_field_model(pretrained=False):
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None)
    model.classifier = nn.Sequential(nn.Linear(960,256),nn.Hardswish(),nn.Dropout(.2),nn.Linear(256,len(FIELD_CLASSES)))
    return model


def letterbox(image):
    return ImageOps.pad(image,(224,224),method=Image.Resampling.BILINEAR,color=(114,114,114))


def field_transform(training=False):
    steps=[transforms.Lambda(letterbox)]
    if training:
        steps += [transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip(),
                  transforms.RandomAffine(20,translate=(.08,.08),scale=(.85,1.1),fill=114),
                  transforms.ColorJitter(.25,.25,.15,.03)]
    return transforms.Compose(steps+[transforms.ToTensor(),transforms.Normalize(IMAGENET_MEAN,IMAGENET_STD)])

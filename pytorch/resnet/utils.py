from PIL import Image
import numpy as np
import torch


def load_image(batch_size):
    img = Image.open("data/ILSVRC2012_val_00000002.JPEG").convert("RGB")
    resized_img = img.resize((224, 224))
    img_data = np.asarray(resized_img).astype("float32")
    img_data = np.transpose(img_data, (2, 0, 1))  # CHW

    # Normalize according to the ImageNet input specification
    imagenet_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    imagenet_stddev = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    norm_img_data = (img_data / 255 - imagenet_mean) / imagenet_stddev

    # Add the batch dimension, as we are expecting 4-dimensional input: NCHW
    img_data = np.expand_dims(norm_img_data, axis=0).repeat(batch_size, axis=0)
    x = torch.from_numpy(img_data).to(torch.float32)

    return x
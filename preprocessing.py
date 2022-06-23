import os
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
import tensorflow_io as tfio
from tensorflow.keras import preprocessing, layers, applications
from tqdm import tqdm


def resample_images(path, path_to, filter_grayscale=True, size=(512, 512)):
    print("Resampling images...")
    os.makedirs(f"{path_to}/grayscale", exist_ok=True)
    os.makedirs(f"{path_to}/colored", exist_ok=True)
    for filename in tqdm(os.listdir(path)):
        img = Image.open(f"{path}/{filename}").convert('RGB')
        img = img.resize(size)
        img_arr = np.array(img).astype(np.float32)/255.
        ab_channel = tfio.experimental.color.rgb_to_lab(img_arr).numpy()[:, :, 1:]
        # remove "noise", or gray images, from our dataset. we want the model to only be exposed to images that have enough color.
        # in the A and B channels, the values are scaled to (-128, 127) where 0 is neutral and has no color.
        colorfulness = 12.8
        if filter_grayscale and np.mean(np.abs(ab_channel)) < colorfulness:
            continue
        ImageOps.grayscale(img).save(f"{path_to}/grayscale/{filename}")
        img.save(f"{path_to}/colored/{filename}")


def extract_image_features(path = "dataset/train"):
    img_model = applications.EfficientNetB1(include_top=False, weights='imagenet')  # we discard the last layer because we only want the features not the classes
    filenames = os.listdir(path)

    print("Extracting image features...")
    for filename in tqdm(filenames):
        if filename.endswith(".jpg"):
            img = Image.open(f"{path}/{filename}")
            img = ImageOps.grayscale(img)   # convert to grayscale
            img = np.array(img).astype(np.float32)
            img = np.stack((img,)*3, axis=-1)  # convert to 3 channel image
            img = tf.constant(img)
            img = tf.expand_dims(img, axis=0)  # add batch dimension
            # img = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(img)
            # print(img.shape)
            img = applications.efficientnet.preprocess_input(img)
            out = img_model(img).numpy()
            # print(out.shape)
            np.save(f"{path}/{filename.split('.')[0]}.npy", out)


if __name__ == "__main__":
    resample_images("dataset/imagenet/train", "dataset/train", filter_grayscale=True)
    resample_images("dataset/imagenet/val", "dataset/val")
    # extract_image_features()
    pass

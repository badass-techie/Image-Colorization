import tensorflow as tf
from tensorflow.keras import Model, Sequential, layers, losses, optimizers, applications
import tensorflow_io as tfio
from tensorflow.lite.python import interpreter
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageOps
import time
import os
from tqdm import tqdm


""" helpers """

#displays time as h:mm:ss
def format_time(seconds):
    return "{}:{:0>2}:{:0>2}".format(int(seconds//3600), int((seconds//60)%60), int(seconds%60))


""" processing the dataset """

train_path = "./dataset/train/"
train_data = os.listdir(train_path)
train_data = list(set([filename.split('.')[0] for filename in train_data]))
train_inputs, train_outputs = [], []

def load_dataset(dataset_size = 8000, batch_size = 32):
    """
        Loads the dataset.
        Images are loaded in the LAB color space.
        The L channel (L for lightness) in the LAB color space is scaled to (0, 100). It can be thought of as a grayscale image.
        The A and B channels are scaled to (-128, 127).
        This function returns only the A and B channels as outputs since that's what the model will be predicting.
        Dataset size has to be a multiple of batch size or it will be truncated.
    """
    print("Loading dataset...")
    dataset_size = dataset_size if  dataset_size < len(train_data) else len(train_data)
    dataset_size = (dataset_size // batch_size) * batch_size
    global train_inputs, train_outputs

    for filename in tqdm(train_data[:dataset_size]):
        img = Image.open(f"{train_path}{filename}.jpg").convert('RGB')
        img = np.array(img).astype(np.float32)/255. # scale between 0 and 1
        img = tfio.experimental.color.rgb_to_lab(img).numpy()[:, :, 1:]
        img = img/128   # rescale a, b channels from (-128, 127) to (-1, 1) (tanh)
        img = np.expand_dims(img, axis=0)  # add concat dimension
        train_outputs.append(img)
        train_inputs.append(np.load(f"{train_path}{filename}.npy"))

    train_inputs, train_outputs = np.concatenate(train_inputs, axis=0), np.concatenate(train_outputs, axis=0)
    train_inputs = np.reshape(train_inputs, [-1, batch_size, train_inputs.shape[-3], train_inputs.shape[-2], train_inputs.shape[-1]])
    train_outputs = np.reshape(train_outputs, [-1, batch_size, train_outputs.shape[-3], train_outputs.shape[-2], train_outputs.shape[-1]])

    print(f"Input shape: {train_inputs.shape}\nOutput shape: {train_outputs.shape}")


test_path = "./dataset/val/"
test_images = np.array(os.listdir(test_path))

def get_batch(batch_size=32):
    """
        Gets a batch from test data. Meant only for inference.
    """
    assert batch_size < test_images.shape[0]
    indices = np.random.choice(test_images.shape[0], batch_size, replace=False)
    grayscale_images = []
    images = []
    for img_filename in test_images[indices]:
        img = Image.open(f"{test_path}{img_filename}").convert('RGB')
        grayscale_image = ImageOps.grayscale(img)
        grayscale_image = np.array(grayscale_image).astype(np.float32)
        grayscale_image = np.stack((grayscale_image,)*3, axis=-1)
        grayscale_image = np.expand_dims(grayscale_image, 0)
        grayscale_images.append(grayscale_image)
        img = np.array(img).astype(np.float32)
        img = np.expand_dims(img, 0)
        images.append(img)

    return np.concatenate(grayscale_images, axis=0), np.concatenate(images, axis=0)



""" model """

features_shape = 14, 14, 1280
model = Sequential(
    [
        layers.Conv2D(320, (3, 3), activation='relu', padding='same', input_shape=features_shape),
        layers.Conv2D(320, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(192, (5, 5), activation='relu', padding='same'),
        layers.Conv2D(192, (5, 5), activation='relu', padding='same'),
        layers.UpSampling2D((2, 2)),
        layers.Conv2D(112, (5, 5), activation='relu', padding='same'),
        layers.Conv2D(112, (5, 5), activation='relu', padding='same'),
        layers.Conv2D(80, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(80, (3, 3), activation='relu', padding='same'),
        layers.UpSampling2D((2, 2)),
        layers.Conv2D(40, (5, 5), activation='relu', padding='same'),
        layers.Conv2D(40, (5, 5), activation='relu', padding='same'),
        layers.UpSampling2D((2, 2)),
        layers.Conv2D(24, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(24, (3, 3), activation='relu', padding='same'),
        layers.UpSampling2D((2, 2)),
        layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.UpSampling2D((2, 2)),
        layers.Conv2D(2, (3, 3), activation='tanh', padding='same'),
    ]
)

class Colorizer(Model):
    """
        End to end model.
    """

    def __init__(self, model):
        super(Colorizer, self).__init__()
        self.feature_extractor = applications.EfficientNetB1(include_top=False, weights='imagenet')
        self.decoder = model

    @tf.function(input_signature=[tf.TensorSpec(shape=[448, 448, 3], dtype=tf.float32)])
    def call(self, inputs):
        inputs = tfio.experimental.color.rgb_to_grayscale(inputs)   # just in case the image is not grayscale
        inputs = tf.repeat(inputs, 3, axis=-1)  # restore 3 channels
        image = tf.expand_dims(inputs, axis=0)     # add batch dimension
        # image = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(image)
        image = applications.efficientnet.preprocess_input(image)
        features = self.feature_extractor(image)
        outputs = self.decoder(features)[0]
        l_channel = (inputs[:, :, :1]/255) * 100 # scale from (0, 255) to (0, 100)
        ab_channel = outputs * 128     # scale from -1,1 to -128, 127
        lab_img = tf.concat([l_channel, ab_channel], axis=-1)   # take L channel from gray image and ab channels from colorized image
        rgb_img = tfio.experimental.color.lab_to_rgb(lab_img)
        rgb_img *= 255  # scale from 0,1 to 0,255
        return rgb_img


""" training """

lr = 1e-3
optimizer = optimizers.Adam(learning_rate=lr)
@tf.function(input_signature=[
    tf.TensorSpec(shape=(None, features_shape[0], features_shape[1], features_shape[2]), dtype=tf.float32),  # (batch_size, seq_len)
    tf.TensorSpec(shape=(None, 448, 448, 2), dtype=tf.float32),
])
def train_step(inputs, labels):
    with tf.GradientTape() as tape:
        logits = model(inputs)
        loss = losses.mean_squared_error(labels, logits)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


def train(num_epochs=200):
    """training loop"""
    loss_history = []
    prev_time = time.time()
    time_elapsed = 0

    # load saved models
    # if os.path.isfile("models/weights.h5"):
    #     model.load_weights("models/weights.h5")

    print("Training...")

    for epoch in range(num_epochs):
        for inp, lbl in tqdm(zip(train_inputs, train_outputs)):
            loss = train_step(inp, lbl)
            loss_history.append(loss.numpy().mean())

            time_elapsed += time.time() - prev_time
            prev_time = time.time()

        print(f"Epoch {epoch + 1}/{num_epochs}. Loss: {loss_history[-1]}. Time elapsed: {format_time(time_elapsed)}\n")
        # save checkpoints
        model.save_weights("models/weights.h5")
        model.save_weights(f"models/epoch{epoch + 1}.h5")

        # plot a graph that will show how our loss varied with time
        plt.plot(loss_history)
        plt.title("Training Progress")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.savefig(os.path.join("./plots/TrainingProgress"))
        # plt.show()
        plt.close()

        gray, rgb = get_batch(5)
        colorizer = Colorizer(model)
        for idx, (gray_img, rgb_img) in enumerate(zip(gray, rgb)):
            os.makedirs(f"outputs/epoch {epoch + 1}", exist_ok=True)
            Image.fromarray(rgb_img.astype(np.uint8)).save(f"outputs/epoch {epoch + 1}/true-{idx}.jpg")
            colorized = colorizer(gray_img).numpy()
            Image.fromarray(colorized.astype(np.uint8)).save(f"outputs/epoch {epoch + 1}/predicted-{idx}.jpg")


""" inference """

def generate_from_saved_weights(num_samples=5):
    for weights in os.listdir("./models"):
        model.load_weights(f"./models/{weights}")
        gray, rgb = get_batch(num_samples)
        colorizer = Colorizer(model)
        for idx, (gray_img, rgb_img) in enumerate(zip(gray, rgb)):
            os.makedirs(f"outputs/{weights}-{idx}", exist_ok=True)
            Image.fromarray(rgb_img.astype(np.uint8)).save(f"outputs/{weights}-{idx}/true-{idx}.jpg")
            colorized = colorizer(gray_img).numpy()
            Image.fromarray(colorized.astype(np.uint8)).save(f"outputs/{weights}-{idx}/predicted-{idx}.jpg")


""" deployment"""

def create_tflite_model():
    colorizer = Colorizer(model)
    converter = tf.lite.TFLiteConverter.from_keras_model(colorizer)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    output = converter.convert()
    open("colorizer.tflite", "wb").write(output)

def run_tflite_model(image):
    tflite_model = open("colorizer.tflite", "rb").read()
    interp = interpreter.Interpreter(model_content=tflite_model)
    interp.allocate_tensors()
    input_details = interp.get_input_details()
    output_details = interp.get_output_details()
    interp.set_tensor(input_details[0]['index'], image)
    interp.invoke()
    output_data = interp.get_tensor(output_details[0]['index'])
    return output_data


if __name__ == "__main__":
    load_dataset()
    # print(train_inputs[0][0])
    # print(train_outputs[0][0])
    # train()
    # generate_from_saved_weights()


    # gray, rgb = get_batch(5)
    # colorizer = Colorizer(model)
    # for idx, (gray_img, rgb_img) in enumerate(zip(gray, rgb)):
    #     os.makedirs(f"outputs/{idx}", exist_ok=True)
    #     Image.fromarray(rgb_img.astype(np.uint8)).save(f"outputs/{idx}/true.jpg")
    #     colorized = colorizer(gray_img).numpy()
    #     Image.fromarray(colorized.astype(np.uint8)).save(f"outputs/{idx}/predicted.jpg")


    # create_tflite_model()
    #
    #
    # img, _ = get_batch(batch_size=1)
    # out = run_tflite_model(img[0])
    # Image.fromarray(out.astype(np.uint8)).save(f"out.jpg")

    pass

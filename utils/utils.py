import numpy as np
import torch
import imageio


def read_image(path):
    return imageio.imread(path)


def image_to_tensor(image):
    return torch.Tensor(np.asarray(image).swapaxes(0, 2)).float()


def normalize_image_tensor(tensor):
    return tensor / 255 - 0.5


def normalized_tensor_to_image(tensor):
    if type(tensor) == torch.Tensor:
        tensor = tensor.numpy()

    if len(tensor.shape) == 4:
        tensor = tensor.swapaxes(1, 3)
    elif len(tensor.shape) == 3:
        tensor = tensor.swapaxes(0, 2)
    else:
        raise Exception("Predictions have shape not in set {3,4}")

    tensor = (tensor + 0.5) * 255
    tensor[tensor > 255] = 255
    tensor[tensor < 0] = 0
    return tensor.round().astype(int)


def to_batch(tensor):
    return tensor.unsqueeze(0)


def read_data(path):
    image = read_image(path)
    tensor = image_to_tensor(image)
    normalized = normalize_image_tensor(tensor)
    return normalized


def process_image(path, model):
    prediction = model.predict(to_batch(read_data(path)))
    prediction_image = normalized_tensor_to_image(prediction).squeeze()
    print(prediction_image.shape)
    imageio.imwrite(path + "sr.jpg", prediction_image)
    return path.split("/")[-1], path.split("/")[-1] + "sr.jpg"

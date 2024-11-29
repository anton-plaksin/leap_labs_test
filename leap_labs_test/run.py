import cv2
import json
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models import resnet50, ResNet50_Weights
from noise_model import NoiseModel


def encode_image(image: np.array) -> np.array:
    'change the axes and scale the image numpy array'
    image = np.transpose(image, axes=(2,1,0)) / 255
    return image


def decode_image(image: np.array) -> np.array:
    'return the initial axes and the scale the image numpy array'
    image = np.clip((255 * image).astype(np.uint16), 0, 255)
    image = np.transpose(image, axes=(2,1,0))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def calculate_init_class_id(class_id: int) -> int:
    'calculate class id in imagenet by class id in imagenette'

    with open("imagenette_classes.json", 'r') as f:
        imagenette_classes = json.load(f)
    weights = ResNet50_Weights.DEFAULT
    imagenette_name = imagenette_classes[class_id]
    imagenet_names = np.array(weights.meta["categories"])

    return np.where(imagenet_names == imagenette_name)[0][0]


def get_noisy_image(init_image: np.array, 
                    init_class_id: int, 
                    lr: float = 1e-3, 
                    epoch_n: int = 1000):
    'add noise to the image such that the classificator make a mistake'

    #define the noise model
    noise = NoiseModel(shape=init_image.shape)

    #define the classification model
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.eval()

    #define preprocess 
    preprocess = weights.transforms()
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=noise.parameters(), lr=lr)

    for _ in range(epoch_n):

        #define noisy image
        noisy_image = torch.tensor(init_image) + noise()

        #predict new class id
        noisy_image = preprocess(noisy_image)
        logits = model(noisy_image.unsqueeze(0))
        new_class_id = logits[0].argmax().item()
        print(new_class_id)

        #if new_class_id=init_class_id make gradient step, else break
        if new_class_id == init_class_id:
            loss = -loss_function(logits, torch.tensor([init_class_id]))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        else:
            break

    return init_image + noise().data.numpy(), new_class_id


def show_results(init_image: np.array, 
                 init_class_id: int, 
                 new_image: np.array, 
                 new_class_id: int):
    'show the initial image and class and the noisy image and new class'

    if (init_image == new_image).all():
        print('The initial classification is incorrect')

    else:
        weights = ResNet50_Weights.DEFAULT
        imagenet_names = np.array(weights.meta["categories"])

        _, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(decode_image(init_image))
        ax1.set_title(f'Initial Class: {imagenet_names[init_class_id]}')

        ax2.imshow(decode_image(new_image))
        ax2.set_title(f'New Class: {imagenet_names[new_class_id]}')

        plt.show()


#run the script
if __name__ == '__main__':

    #parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_path", required=True, type=str)
    ap.add_argument("--class_id", required=True, type=int)
    args = vars(ap.parse_args())

    #load initial image and initial class id according to imagenet
    init_image = encode_image(cv2.imread(args["image_path"]))
    init_class_id = calculate_init_class_id(args["class_id"])

    #apply some noise to the initial image to change their class
    new_image, new_class_id = get_noisy_image(init_image=init_image, 
                                              init_class_id=init_class_id)


    #show the results
    show_results(init_image=init_image, 
                 init_class_id=init_class_id, 
                 new_image=new_image, 
                 new_class_id=new_class_id)

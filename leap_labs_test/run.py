import cv2
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.models import resnet50, ResNet50_Weights
from noise_model import NoiseModel


def encode_image(image: np.array) -> torch.tensor:
    'Change the axes, scale the image numpy array, and transfer it to the tensor'
    image = np.transpose(image, axes=(2,1,0)) / 255
    return torch.tensor(image)


def decode_image(image: torch.tensor) -> np.array:
    'Return the initial axes and the scale the image numpy array'
    image = np.clip((255 * image.data.numpy()).astype(np.uint16), 0, 255)
    image = np.transpose(image, axes=(2,1,0))
    return image


def get_noisy_image(init_image: np.array, 
                    target_class: int, 
                    lr: float, 
                    epoch_n: int, 
                    noise_coef: int):
    'Add noise to the image such that the classificator make a mistake'

    #encode initial image
    init_image = encode_image(init_image)

    #define the noise model
    noise = NoiseModel(shape=init_image.shape)

    #define the classification model
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.eval()

    #define preprocess, loss_function, optimizer
    preprocess = weights.transforms()
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=noise.parameters(), lr=lr)

    #find the initial class
    logits = model(preprocess(init_image).unsqueeze(0))
    init_class = logits[0].argmax().item()

    for _ in tqdm(range(epoch_n)):

        #define the noisy image
        noisy_image = init_image + noise()

        #predict the new class
        noisy_image = preprocess(noisy_image)
        logits = model(noisy_image.unsqueeze(0))
        new_class = logits[0].argmax().item()

        #if new_class = target_class break else make a gradient step
        if new_class == target_class:
            break

        else:
            loss = loss_function(logits, torch.tensor([target_class])) + noise_coef * torch.mean((noise()) ** 2)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    #decode noisy image
    new_image = decode_image(init_image + noise())
    return init_class, new_image, new_class


def show_results(init_image: np.array, 
                 init_class: int, 
                 new_image: np.array, 
                 target_class: int, 
                 new_class:int):
    'Show the initial image and class and the noisy image and new class'

    if target_class != new_class:
        print('Warring! The target class not reached, increase number of epoch')

    weights = ResNet50_Weights.DEFAULT
    imagenet_names = np.array(weights.meta["categories"])

    _, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(cv2.cvtColor(init_image, cv2.COLOR_BGR2RGB))
    ax1.set_title(f'Initial Class: {imagenet_names[init_class]}')

    ax2.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
    ax2.set_title(f'New Class: {imagenet_names[new_class]}')

    plt.show()


#run the script
if __name__ == '__main__':

    #parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_path", required=True, type=str)
    ap.add_argument("--target_class", required=True, type=int)
    ap.add_argument("--epoch_n", default=250, type=int)
    ap.add_argument("--lr", default=1e-3, type=float)
    ap.add_argument("--noise_coef", default=1e6, type=float)
    args = vars(ap.parse_args())

    #load initial image and target_class
    init_image = cv2.imread(args["image_path"])
    target_class = args["target_class"]

    #add some noise to the initial image to change their class to target one
    init_class, new_image, new_class = get_noisy_image(init_image=init_image, 
                                                       target_class=target_class, 
                                                       lr=args["lr"], 
                                                       epoch_n=args["epoch_n"], 
                                                       noise_coef=args["noise_coef"])


    #show the results
    show_results(init_image=init_image, 
                 init_class=init_class, 
                 new_image=new_image, 
                 target_class=target_class, 
                 new_class=new_class)

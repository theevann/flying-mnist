# Flying MNIST Database

![flying_mnist.gif](flying_mnist.gif)

This PyTorch dataset class generates a flying colored MNIST dataset and corresponding dense labels.
This dataset has been made for segmentation purposes.

The dataset is generated on the fly with images randomly sampled in the provided input MNIST dataset.
Querying twice the same index will yield the same sample.

### Usage
```python
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from flying_mnist import FlyingMNIST

train_dataset = MNIST(
    root='./data',
    train=True,
    download=True,
    transform=ToTensor()
)

flying_mnist = FlyingMnist(train_dataset, seq_length=10, min_img_num=3, max_img_num=7, img_size=(200, 200), rotation=20)

sample, label = flying_mnist[0]
# sample.shape: 10 x 3 x 200 x 200
# label.shape: 10 x 200 x 200

# Check generated sample by creating a gif:
flying_mnist.tensor2gif(sample, duration=100, save=True)
```


### Parameters

 - `dataset`: PyTorch MNIST dataset (train or val)
 - `seq_length`: Number of frames to generate (Default:20)
 - `img_size`: Size of generated image (Default:(100, 100))
 - `min_img_num`: Minimum number of digit to draw (Default:1)
 - `max_img_num`: Maximum number of digit to draw (Default:As `min_img_num`)
 - `rotation`: If rotation is non-zero, digits will be rotated while moving. Initial and final angle will be between (-`rotation`,`rotation`) (Default:0)
 - `colorize`: If True, draw each number with a different random color (up to 7 different colors) (Default:True)
 - `ignore_index`: Which number to fill the black space with in the generated labels (Default:-1)

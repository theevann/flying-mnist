import random
from io import BytesIO

import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset


class FlyingMnist(Dataset):
    def __init__(self, dataset, seq_length=20, img_size=(100, 100), min_img_num=1, max_img_num=None, rotation=0, colorize=True, ignore_index=-1):
        self.dataset = dataset
        self.seq_length = seq_length
        self.img_size = img_size
        self.min_img_num = min_img_num
        self.max_img_num = max_img_num or min_img_num
        self.rotation = rotation
        self.ignore_index = ignore_index

        if colorize:
            self.colors = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1], [.5, .5, .5]]).view(-1, 3, 1, 1)
            if self.max_img_num > len(self.colors):
                print("Ouch, not enough different colors")
        else:
            self.colors = torch.ones(self.max_img_num, 3, 1, 1)

    def tensor2gif(self, x, duration=100, save=False):
        from IPython.display import Image
        imgs = self.tensor2imgs(x)
        if save:
            imgs[0].save("flying_mnist.gif", save_all=True, append_images=imgs[1:], duration=duration, loop=0)
        fake_file = BytesIO()
        imgs[0].save(fake_file, "gif", save_all=True, append_images=imgs[1:], duration=duration, loop=0)
        return Image(fake_file.getvalue())

    def tensor2imgs(self, x):
        return list(map(TF.to_pil_image, x))

    def __getitem__(self, x):
        random.seed(x)
        num_samples = random.randint(self.min_img_num, self.max_img_num)
        samples_indices = random.sample(list(range(0, len(self.dataset))), num_samples)
        samples, labels = zip(*[self.dataset[i] for i in samples_indices])
        pil_samples = list(map(TF.to_pil_image, samples))

        torch.manual_seed(x)
        max_size = torch.tensor(samples[0].shape[-2:], dtype=float).norm().ceil().int().item()
        xs = torch.randint(0, self.img_size[1]-max_size, (num_samples, 2))
        ys = torch.randint(0, self.img_size[0]-max_size, (num_samples, 2))
        rs = torch.randint(-self.rotation, self.rotation, (num_samples, 2)) if self.rotation != 0 else torch.zeros_like(ys)

        imgs = torch.zeros(self.seq_length, 3, *self.img_size)
        lbls = torch.empty(self.seq_length, *self.img_size, dtype=int).fill_(self.ignore_index)
        for i in range(self.seq_length):
            a = 1. * i / self.seq_length
            weights = torch.tensor([1-a, a])
            img = TF.to_pil_image(imgs[i])

            for j in range(num_samples):
                pil_sample = pil_samples[j]
                x = xs[j].mul(weights).sum().int().item()
                y = ys[j].mul(weights).sum().int().item()
                r = rs[j].mul(weights).sum().int().item()

                pil_sample = TF.rotate(img=pil_sample, angle=r, expand=True)
                pil_sample_color = self._colorize(pil_sample, j)
                pil_mask = pil_sample.convert('L').point(self._thresholf_fn, mode='1')
                img.paste(pil_sample_color, (x, y), mask=pil_mask)

                tens_mask = TF.to_tensor(pil_mask)[0]
                lbls[i, y:y+tens_mask.size(0), x:x+tens_mask.size(1)][tens_mask > 0] = labels[j]

            imgs[i] = TF.to_tensor(img)

        return imgs, lbls

    def _thresholf_fn(self, x):
        return 255 if x > 100 else 0

    def _colorize(self, pil_img, color_idx):
        tens_img = TF.to_tensor(pil_img)
        tens_img = tens_img.expand((3, *tens_img.shape[-2:])) * self.colors[color_idx % len(self.colors)]
        return TF.to_pil_image(tens_img)


if __name__ == "__main__":
    from torchvision.datasets import MNIST
    from torchvision.transforms import ToTensor, Compose, RandomAffine

    train_dataset = MNIST(
        root='./data',
        train=True,
        download=True,
        transform=Compose((
            RandomAffine(degrees=0, scale=[1., 1.5]),
            ToTensor()
        )),
    )

    flying_mnist = FlyingMnist(train_dataset, seq_length=10, min_img_num=3, max_img_num=7, img_size=(200, 200), rotation=20)
    sample, label = flying_mnist[0]
    
    print(sample.shape, label.shape)
    flying_mnist.tensor2gif(sample, duration=100, save=True)


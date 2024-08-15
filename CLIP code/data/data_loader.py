import torchvision.datasets as datasets
from torchvision import transforms
import os
import clip

from PIL import ImageFile, Image

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageTextData(object):
    def __init__(self, dataset, root, preprocess, prompt='a picture of a'):
        if type(dataset) is int:
            dataset = self._DATA_FOLDER[dataset]
        dataset = os.path.join(root, dataset)
        if dataset == 'imagenet-r':
            data = datasets.ImageFolder(
                'imagenet-r', transform=self._TRANSFORM)
            labels = open('imagenetr_labels.txt').read().splitlines()
            labels = [x.split(',')[1].strip() for x in labels]
        else:
            data = datasets.ImageFolder(dataset, transform=self._TRANSFORM)
            labels = data.classes
        self.data = data
        self.labels = labels
        if prompt:
            self.labels = [prompt + ' ' + x for x in self.labels]

        self.preprocess = preprocess
        self.text = clip.tokenize(self.labels)

    def __getitem__(self, index):
        image, label = self.data.imgs[index]
        if self.preprocess is not None:
            image = self.preprocess(Image.open(image))
        text_enc = self.text[label]
        return image, text_enc, label

    def __len__(self):
        return len(self.data)

    @staticmethod
    def get_data_name_by_index(index):
        name = ImageTextData._DATA_FOLDER[index]
        name = name.replace('/', '_')
        return name

    _DATA_FOLDER = [
        'dataset/OfficeHome/Art',  # 0
        'dataset/OfficeHome/Clipart',
        'dataset/OfficeHome/Product',
        'dataset/OfficeHome/RealWorld',

        'dataset/imagenet-r/',  # 4

        'dataset/office31/amazon',  # 5
        'dataset/office31/webcam',
        'dataset/office31/dslr',

        'dataset/BT/train',  # 8
        'dataset/BT/valid',

        'dataset/HAM10000/train',  # 10
        'dataset/HAM10000/test',

        'dataset/Glaucoma/train',  # 12
        'dataset/Glaucoma/test',
        'dataset/Glaucoma/val',

        'dataset/ChestXray/train', # 15
        'dataset/ChestXray/test',

        'dataset/Adaptiope/R',  # 17
        'dataset/Adaptiope/P',
        
        'dataset/NoisyDA/gaussian_noise1',  # 19
        'dataset/NoisyDA/gaussian_noise2',  # 20
        'dataset/NoisyDA/gaussian_noise3',  # 21
        'dataset/NoisyDA/gaussian_noise4',  # 22
        'dataset/NoisyDA/gaussian_noise4',  # 23
        'dataset/NoisyDA/global',       # 24

        'dataset/Eye_disease/train',  # 25
        'dataset/Eye_disease/test',
    ]

    _TRANSFORM = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


if __name__ == '__main__':
    print(ImageTextData.get_data_name_by_index(20))

"""Read instructor image dictionaries and retain their train/test partition."""
from pathlib import Path
import pickle
import numpy as np
from local_code.base_class.dataset import dataset


class Dataset_Loader(dataset):
    def __init__(self, data_dir, name):
        super().__init__(name, 'Instructor-provided image classification data')
        if name not in ('MNIST', 'ORL', 'CIFAR'):
            raise ValueError(name)
        self.name = name
        self.data_dir = Path(data_dir)
        self.classes = 40 if name == 'ORL' else 10
        self.label_offset = 1 if name == 'ORL' else 0

    def load(self):
        # Only use the trusted instructor dataset, never arbitrary pickle files.
        with (self.data_dir / self.name).open('rb') as f:
            raw = pickle.load(f)
        output = {}
        for split in ('train', 'test'):
            images, labels = [], []
            for record in raw[split]:
                image = np.asarray(record['image'])
                expected = {'MNIST': (28,28), 'ORL': (112,92,3), 'CIFAR': (32,32,3)}[self.name]
                if image.shape != expected or not np.isfinite(image).all():
                    raise ValueError(f'{self.name}/{split}: invalid image shape or values')
                if image.min() < 0 or image.max() > 255 or not np.equal(image, image.astype(np.uint8)).all():
                    raise ValueError('Expected integer pixels in [0,255]')
                if self.name == 'ORL':
                    if not (np.array_equal(image[:,:,0],image[:,:,1]) and np.array_equal(image[:,:,0],image[:,:,2])):
                        raise ValueError('Expected identical ORL channels')
                    image = image[:,:,0][None,:,:]
                elif self.name == 'MNIST':
                    image = image[None,:,:]
                else:
                    image = image.transpose(2,0,1)
                label = int(record['label']) - self.label_offset
                if label < 0 or label >= self.classes:
                    raise ValueError('Invalid class label')
                images.append(image.astype(np.uint8))
                labels.append(label)
            output[split] = {'X': np.ascontiguousarray(np.stack(images)),
                             'y': np.asarray(labels, dtype=np.int64)}
        return output

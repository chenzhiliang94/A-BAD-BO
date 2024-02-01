#
# This is a sample Notebook to demonstrate how to read "MNIST Dataset"
#
import numpy as np # linear algebra
import struct
from array import array
from os.path  import join

#pytorch utility imports
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset

#
# MNIST Data Loader Class
#
class MnistDataloader(object):
    def __init__(self, training_images_filepath,training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath
    
    def read_images_labels(self, images_filepath, labels_filepath):        
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())        
        
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())        
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img            
        
        return images, labels
            
    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train),(x_test, y_test) 

def generate_data_loader(num_datapts, batch_size, start_idx, label_int):
    input_path = 'mnist'
    training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
    training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
    test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
    test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')
    
    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()
    
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    y_train = 1.0*(y_train == label_int) # mask to binary classifier
    #local dataset # hack
    train_images_tensor = torch.tensor(x_train)/255.0
    train_labels_tensor = torch.LongTensor(y_train)
    local_train_tensor = TensorDataset(train_images_tensor[start_idx:start_idx+num_datapts], train_labels_tensor[start_idx:start_idx+num_datapts])
    system_train_tensor = TensorDataset(train_images_tensor[2000:2000+num_datapts], train_labels_tensor[2000:2000+num_datapts])


    #val
    val_images_tensor = torch.tensor(x_test)/255.0
    val_labels_tensor = torch.tensor(y_test)
    val_tensor = TensorDataset(val_images_tensor, val_labels_tensor)
    
    local_train_loader = DataLoader(local_train_tensor, batch_size=batch_size, num_workers=2, shuffle=False)
    system_train_loader = DataLoader(system_train_tensor, batch_size=batch_size, num_workers=2, shuffle=False)
    val_loader = DataLoader(val_tensor, batch_size=batch_size, num_workers=2, shuffle=False)
    
    return local_train_loader, system_train_loader
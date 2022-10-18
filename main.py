import cv2
import numpy as np
import torch
import torchvision

import BasketballActionRecognitionModel from BasketballActionRecognitionModel

def output_library_versions():
    print('OpenCV Version: ', cv2.__version__)
    print('Numpy Version: ', np.__version__)
    print('PyTorch Version: ', torch.__version__)
    print('Torchvision Version: ', torchvision.__version__)

if __name__ == '__main__':
    # Output library versions
    output_library_versions()

    model = BasketballActionRecognitionModel()




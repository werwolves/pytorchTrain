import numpy as np
import cv2
from PIL import Image

class KeepKeys(object):
    def __init__(self, keep_keys, **kwargs):
        self.keep_keys = keep_keys

    def __call__(self, data):
        data_list = []
        for key in self.keep_keys:
            data_list.append(data[key])
        return data_list

class DecodeImage:
    """Decode image from bytes."""
    def __init__(self,
                 img_mode='RGB',
                 channel_first = False,
                 ignore_orientation = False,
                 **kwargs):

        self.img_mode = img_mode
        self.channel_first = channel_first
        self.ignore_orientation = ignore_orientation

    def __call__(self, data):
        img = data['image']
        img = np.frombuffer(img, np.uint8)
        if self.ignore_orientation:
            img = cv2.imdecode(img, cv2.IMREAD_IGNORE_ORIENTATION | cv2.IMREAD_COLOR) 
        else:
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    
        if img is None:
            return None
        
        if self.img_mode == 'GRAY':
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
        elif self.img_mode == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif self.img_mode == 'BGR':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if self.channel_first:
            img = np.transpose(img, (2, 0, 1))
        
        data['image'] = img
        return data
import numpy as np
import imgaug
import imgaug.augmenters as iaa

class AugmentBuilder:
    def __init__(self):
        pass
    def build(self, args, root=True):
        if args is None or len(args) == 0:
            return None
        if isinstance(args, list):
            if root:
                sequence = [self.build(aug, root=False) for aug in args] # args = [{},{},...{}]
                return iaa.Sequential(sequence)
            else:
                ...
        elif isinstance(args, dict):
            aug_type = args.pop('type')
            aug_args = args.pop('args')
            if isinstance(aug_args, list):
                aug_args = tuple(aug_args)
            if hasattr(iaa, aug_type):
                aug_class = getattr(iaa, aug_type)
                return aug_class(**aug_args)
            else:
                raise ValueError(f"Unknown augmentation type: {aug_type}")        
        else:
            raise ValueError("Augmentation should be a list or a dictionary.")
        
        
class IaaAugmenter:
    def __init__(self, augmentations=None):
        """
        Initialize the IaaAugmenter with a list of augmentations.

        :param augmentations: List of imgaug augmenters to apply.
        """
        if augmentations is None:
            augmentations = [
                    {
                    'type': 'Fliplr',
                    'args': {
                                'p': 0.5
                            }
                    }, 
                    {
                        'type': 'Affine',
                        'args': {
                            'rotate': [-10, 10]
                        }
                    }, 
                    {
                    'type': 'Resize',
                    'args': {
                        'size': [0.5, 3]
                    }
                    }
                ]
            self.augmenter = AugmentBuilder().build(augmentations)
    def may_augment_annotation(self, aug, data, shape):
        if aug is None:
            return data
        line_poly = []
        for poly in data["polys"]:
            new_poly = self.may_augment_poly(aug, shape, poly)
            line_poly.append(new_poly)
        data["polys"] = line_poly
        return data
    
    def may_augment_poly(self, aug, im_shape, poly):
        # 这不就体现了 imgaug 的强大之处吗？ ！！！！ 
        keypoints = [imgaug.Keypoint(x=point[0], y=point[1]) for point in poly]
        keypoints_on_image = imgaug.KeypointsOnImage(keypoints, shape=im_shape)
        keypoints_on_image_aug = aug.augment_keypoints([keypoints_on_image])[0]
        
        new_keypoints = keypoints_on_image_aug.keypoints
        new_poly = [[kp.x, kp.y] for kp in new_keypoints]
        return new_poly
        
        
    def __call__(self, data):
        img = data['image']
        shape = img.shape

        if self.augmenter is not None:
            aug = self.augmenter.to_deterministic()
            data['image'] = aug.augment_image(img)
            
            
            
            
            
            
            
            
            
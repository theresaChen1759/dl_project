import os 
import geopandas as gpd 
import pandas as pd
import pickle
import sys
import numpy as np 
from PIL import Image
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, cast
import random
import pickle

import torch
from torchvision.datasets.vision import VisionDataset
from torch.utils.data import Dataset

Image.MAX_IMAGE_PIXELS = None

def make_triplet_dataset_cropped(directory: str) -> List[Tuple[str,str,str]]:
    instances = []
    root = os.path.expanduser(directory)
    subdirectories = os.listdir(directory)
    subdirectories = subdirectories[:100] #for speed, take out later
    clean_subdirectories = [clean_topomap_name(x) for x in subdirectories]
    for index, subdirectory in enumerate(subdirectories):
        area_name = clean_topomap_name(subdirectory)
        matching_maps = []
        for index2, map2 in enumerate(clean_subdirectories): #let's find all the matching maps
            if index2 == index:
                continue
            else:
                if map2 == area_name:
                    matching_maps.append(subdirectories[index2])
        if matching_maps == 0: #if no duplicates then skip
            continue
        sub_root = os.path.join(root, subdirectory)
        sub_files = os.listdir(sub_root)
        for file in sub_files:
            anchor = os.path.join(sub_root, file)
            for matching_map in matching_maps:
                positive_root = os.path.join(root, matching_map)
                positive = os.path.join(positive_root, file)
                negative_path = random.choice(subdirectories)
                while clean_topomap_name(negative_path) == area_name: #make sure neg is actually neg
                    negative_path = random.choice(subdirectories)
                negative_root = os.path.join(root, negative_path)
                negative_file = random.choice(os.listdir(negative_root))
                negative = os.path.join(negative_root, negative_file)
                triplet = (anchor, positive, negative)
                instances.append(triplet)
    return instances

def make_triplet_dataset(directory: str) -> List[Tuple[str,str,str]]: 
    '''
    This if for just straight up resizing the images, based on the image filenames
    '''
    instances = []
    root = os.path.expanduser(directory)
    files = os.listdir(directory)
    clean_files = [clean_topomap_name(x) for x in files]
    for index, file in enumerate(files):
        area_name = clean_topomap_name(file)
        matching_maps = []
        for index2, map2 in enumerate(clean_files): #let's find all the matching maps
            if index2 == index:
                continue
            else:
                if map2 == area_name:
                    matching_maps.append(files[index2])
        if matching_maps == 0: #if no duplicates then skip
            continue
        anchor = os.path.join(root, file)
        for matching_map in matching_maps:
            positive = os.path.join(root, matching_map)
            negative_path = random.choice(files)
            while clean_topomap_name(negative_path) == area_name: #make sure neg is actually neg
                negative_path = random.choice(files)
            negative = os.path.join(root, negative_path)
            triplet = (anchor, positive, negative)
            instances.append(triplet)
    return instances

def make_triplet_dataset_from_dict_topo(dict_path: str, root_dir: str, extension = '.tif'
    ) -> List[Tuple[str,str,str]]:
    '''
    Assuming pairs are already in dict, for resizing images method
    :param dict_path: should point to a pickle dict in the format (map_name, list(positive_pair_map_names))
        the map_name should be in pure format (aka just the filename, no root, no extension)
    :root_dir: should point to a pickle dict in the format (map_name, list(positive_pair_map_names))
        the map_name should be in pure format (aka just the filename, no root, no extension)
    '''
    instances = []
    root = os.path.expanduser(root_dir)
    files = os.listdir(root_dir)
    # files = files[:100] #for speed, take out later

    with open(dict_path, 'rb') as handle:
        dict_pos = pickle.load(handle)

    for index, file in enumerate(files):
        anchor_name = file.split('/')[-1].split('.')[0]
        full_anchor_file = os.path.join(root, file)
        try:
            positives = dict_pos[anchor_name]
        except:
            print(f'Cannot train on {anchor_name} due to no positive pairs')
            continue
        for positive_file in positives:
            real_filename = positive_file + extension
            full_positive_path = os.path.join(root, real_filename)
            #check that positive path exists in our root
            if not os.path.exists(full_positive_path):
                print(f'Positive file {real_filename} does not exist in root directory')
                continue

            #choose a random negative 
            negative_file = random.choice(files)
            while negative_file.split('/')[-1].split('.')[0] in positives:
                negative_file = random.choice(files)
            full_negative_file = os.path.join(root, negative_file)
            triplet = (full_anchor_file, full_positive_path, full_negative_file)
            instances.append(triplet)
    return instances

def make_triplet_dataset_from_dict_geo(dict_path: str, geo_root_dir: str, topo_root_dir: str, 
extension = '.tif'
    ) -> List[Tuple[str,str,str]]:
    '''
    Assuming pairs are already in dict, for resizing images method
    :param dict_path: should point to a pickle dict in the format (map_name, list(positive_pair_map_names))
        the map_name should be in pure format (aka just the filename, no root, no extension)
    :root_dir: should point to a pickle dict in the format (map_name, list(positive_pair_map_names))
        the map_name should be in pure format (aka just the filename, no root, no extension)
    '''
    instances = []
    geo_root = os.path.expanduser(geo_root_dir)
    geo_files = os.listdir(geo_root)
    topo_files = os.listdir(topo_root_dir)
    # files = files[:100] #for speed, take out later

    with open(dict_path, 'rb') as handle:
        dict_pos = pickle.load(handle)
    count = 0
    for index, file in enumerate(geo_files):
        anchor_name = file.split('/')[-1].split('.')[0]
        full_anchor_file = os.path.join(geo_root, file)
        try:
            positives = dict_pos[anchor_name]
        except:
            print(f'Cannot train on {anchor_name} due to no positive pairs')
            count += 1
            continue
        for positive_file in positives:
            real_filename = positive_file + extension
            full_positive_path = os.path.join(topo_root_dir, real_filename)
            #check that positive path exists in our root
            if not os.path.exists(full_positive_path):
                print(f'Positive file {real_filename} does not exist in root directory')
                continue

            #choose a random negative 
            negative_file = random.choice(topo_files)
            while negative_file.split('/')[-1].split('.')[0] in positives:
                negative_file = random.choice(topo_files)
            full_negative_file = os.path.join(topo_root_dir, negative_file)
            triplet = (full_anchor_file, full_positive_path, full_negative_file)
            instances.append(triplet)
    return instances

def create_target_dict(sample_paths):
    targets_dict = {}
    for sample in sample_paths:
        #find all duplicates
        duplicates = []
        map_name = sample.split('/')[-1]
        map_name = clean_topomap_name(map_name)
        for other_maps in sample_paths:
            other_map_name = other_maps.split('/')[-1]
            other_map_name = clean_topomap_name(other_map_name)
            if map_name == other_map_name:
                duplicates.append(other_maps)
        targets_dict[sample] = duplicates
    return targets_dict

def clean_topomap_name(name: str) -> str:
    name_parts = name.split('_')
    area_name = name_parts[0] + "_" + name_parts[1]
    return area_name

def pil_loader(path: str) -> Image.Image:
    im_path = path
    img = Image.open(im_path)
    return img

# make_dataset('/scratch.global/chen7924/MN_Topo_Crop')
class TripletDataset(VisionDataset):
    def __init__(
        self,
        root: str,
        geo_root: str = None,
        positives_dict: str = None,
        anchor_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        super(TripletDataset, self).__init__(root, transform=anchor_transform,
                                            target_transform=target_transform)
        self.anchor_transform = anchor_transform
        self.target_transform = target_transform
        
        if positives_dict is not None: #this should be the default
            if geo_root is not None:
                self.samples = make_triplet_dataset_from_dict_geo(positives_dict, geo_root, root)
                print('Number of training samples: ', len(self.samples))
            else:
                self.samples = make_triplet_dataset_from_dict_topo(positives_dict, root)
        else:
            self.samples = make_triplet_dataset(root)
        self.cache = {}

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        anchor_path, positive_path, negative_path = self.samples[index]
        anchor = pil_loader(anchor_path)
        positive = pil_loader(anchor_path)
        negative = pil_loader(anchor_path)

        if self.anchor_transform is not None:
            anchor = self.transform(anchor)
        if self.target_transform is not None:
            positive = self.target_transform(positive)
            negative = self.target_transform(negative)

        return anchor, positive, negative

    def __len__(self) -> int:
        return len(self.samples)

class ImageDataset(VisionDataset):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        valid_files_list=None,
    ) -> None:
        super(ImageDataset, self).__init__(root, transform=transform)
        self.transform = transform
        if valid_files_list is not None:
            # with open(valid_files_path, 'rb') as handle:
            #     valid_keys = pickle.load(handle).keys()
            #     self.valid_files = [y + '.tif' for y in valid_keys]
            self.valid_files = valid_files_list
            self.samples = sorted([os.path.join(root, x) for x in os.listdir(root) if x in self.valid_files])
        else:
            self.samples = sorted([os.path.join(root, x) for x in os.listdir(root)])
        # self.targets_dict = create_target_dict(self.samples)
        self.cache = {}

    def __getitem__(self, index: int) -> Tuple[Any, Any, Any]:
        image_path = self.samples[index]
        image = pil_loader(image_path)

        if self.transform is not None:
            image = self.transform(image)

        return image, image_path

    def __len__(self) -> int:
        return len(self.samples)
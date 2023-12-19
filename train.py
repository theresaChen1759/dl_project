from utils.logger import WandbLogger, MetricLogger
from utils.dataset_loader import TripletDataset, ImageDataset, clean_topomap_name
from utils.preprocess import normal_transform
from utils.criterion import triplet_loss

import numpy as np
import argparse
import sys
import os
import yaml
import pickle
from sklearn.metrics.pairwise import euclidean_distances
from einops import rearrange
import pickle

import torch
from transformers import (AutoImageProcessor, ViTMAEConfig, ViTMAEModel,
                            ResNetModel, ViTModel,
                            ResNetConfig)
from transformers.trainer_utils import get_last_checkpoint
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import AdamW
import torchvision.models as models
import torch.nn.functional as F

def main(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if args.model_type == 'resnet':
        model_config = ResNetConfig()
        model = ResNetModel.from_pretrained("microsoft/resnet-50")
    else:
        model_config = ViTMAEConfig()
        if args.checkpoint_path is not None:
            model = ViTMAEModel.from_pretrained(args.checkpoint_path)
        else:
            model = ViTMAEModel.from_pretrained("facebook/vit-mae-base")
    model.to(device)
    
    if args.topo_only:
        train_dataset = TripletDataset(args.topo_train_input_path, positives_dict = args.positives_dict_path,
                            anchor_transform = normal_transform, target_transform = normal_transform)
        topo_dataset = ImageDataset(args.topo_full_input_path,
                            transform = normal_transform, valid_files_path = args.positives_dict_path)
    else:
        with open(args.positives_dict_path, 'rb') as handle:
            match_dict = pickle.load(handle)
            valid_keys = match_dict.keys()
            flattened_values = [item for sublist in match_dict.values() for item in sublist]
            valid_geo_files = [y + '.tif' for y in valid_keys]
            valid_topo_files = [z + '.tif' for z in flattened_values]
        train_dataset = TripletDataset(args.topo_full_input_path, geo_root = args.geo_train_input_path, 
                                positives_dict = args.positives_dict_path,
                                anchor_transform = normal_transform, target_transform = normal_transform)
        topo_dataset = ImageDataset(args.topo_full_input_path,
                                transform = normal_transform, valid_files_list = valid_topo_files)
        geo_dataset = ImageDataset(args.geo_test_input_path,
                                transform = normal_transform, valid_files_list = valid_geo_files) 
      
    train_indices, val_indices = split_train_val(train_dataset)

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, 
                                           sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                            sampler=val_sampler)
    topo_loader = torch.utils.data.DataLoader(topo_dataset, batch_size=args.batch_size)
    if not args.topo_only:
        geo_loader = torch.utils.data.DataLoader(geo_dataset, batch_size=args.batch_size)
    else:
        testset = os.listdir(args.test_input_path)

    optimizer = AdamW(model.parameters(), lr = args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    if args.log_wandb:
        log_writer = WandbLogger(args)
        num_training_steps_per_epoch = len(train_loader) 

    if not os.path.exists(args.model_save_path):
        os.mkdir(args.model_save_path)

    if args.eval_only:
        args.start_epoch = args.epochs
    best_val_loss = float('inf')
    for epoch in range(args.start_epoch, args.epochs):
        if args.log_wandb:
            log_writer.set_step(epoch * num_training_steps_per_epoch)
        else:
            log_writer = None
        train_stats = train_one_epoch(train_loader, 
                model, optimizer, device,
                args, log_writer, model_base = args.model_type)
        val_stats = train_one_epoch(val_loader, 
                model, optimizer, device,
                args, log_writer, val_flag = True, 
                model_base = args.model_type)
        train_loss = train_stats['[Epoch] train_loss']
        val_loss = val_stats['[Epoch] val_loss']
        scheduler.step(val_loss)
        print(f'Epoch: {epoch}, Training Loss: {train_loss}, Validation Loss: {val_loss}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(args.model_save_path, 'best-val_model')
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            model.save_pretrained(save_path)

    print('Beginning Evaluation')
    if args.model_type == 'vit-mae':
        best_pt = os.path.join(args.model_save_path, 'best-val_model')
        if not os.path.exists(best_pt):
            model = ViTModel.from_pretrained('facebook/vit-mae-base').to(device)
        else:
            model = ViTModel.from_pretrained(best_pt).to(device)
    else:
        best_pt = os.path.join(args.model_save_path, 'best-val_model')
        if not os.path.exists(best_pt):
            model = ResNetModel.from_pretrained("microsoft/resnet-50").to(device)
        else:
            model = ResNetModel.from_pretrained(best_pt).to(device)
    if args.topo_only:
        evaluate_topo(topo_loader, model, model_config, args.positives_dict_path, device,
                    testset, len(testset), args.batch_size, len(topo_dataset), args.results_save_path,
                    embeddings_array = args.topo_distances_mtx_path, top_k = args.top_k)
    else:
        evaluate_geo_topo(topo_loader, geo_loader, model, model_config, args.positives_dict_path, device,
                            args.batch_size, len(topo_dataset), len(geo_dataset), args.results_save_path,
                            topo_embeddings_array = args.topo_distances_mtx_path, geo_embeddings_array = args.geo_distances_mtx_path,
                            top_k = args.top_k, model_base = args.model_type)

def train_one_epoch(train_loader, 
            model, optimizer, device,
            args, log_writer, val_flag = False, model_base = 'vit-mae'):
    for step, (anchor, positive, negative) in enumerate(train_loader):
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)
        if val_flag:
            model.eval()
            with torch.no_grad():
                if model_base == 'vit-mae':
                    anchor_outputs = model(anchor).last_hidden_state
                    anchor_outputs = anchor_outputs[:, -1, :]
                    positive_outputs = model(positive).last_hidden_state
                    positive_outputs = positive_outputs[:, -1, :]
                    negative_outputs = model(negative).last_hidden_state
                    negative_outputs = negative_outputs[:, -1, :]
                else:
                    anchor_outputs = model(anchor).last_hidden_state
                    anchor_outputs = F.adaptive_avg_pool2d(anchor_outputs, (1, 1)).view(anchor_outputs.size(0), -1)
                    positive_outputs = model(positive).last_hidden_state
                    positive_outputs = F.adaptive_avg_pool2d(positive_outputs, (1, 1)).view(positive_outputs.size(0), -1)
                    negative_outputs = model(negative).last_hidden_state
                    negative_outputs = F.adaptive_avg_pool2d(negative_outputs, (1, 1)).view(negative_outputs.size(0), -1)
        else:
            model.train()
        
            if model_base == 'vit-mae':
                    anchor_outputs = model(anchor).last_hidden_state
                    anchor_outputs = anchor_outputs[:, -1, :]
                    positive_outputs = model(positive).last_hidden_state
                    positive_outputs = positive_outputs[:, -1, :]
                    negative_outputs = model(negative).last_hidden_state
                    negative_outputs = negative_outputs[:, -1, :]
            else:
                anchor_outputs = model(anchor).last_hidden_state
                anchor_outputs = F.adaptive_avg_pool2d(anchor_outputs, (1, 1)).view(anchor_outputs.size(0), -1)
                positive_outputs = model(positive).last_hidden_state
                positive_outputs = F.adaptive_avg_pool2d(positive_outputs, (1, 1)).view(positive_outputs.size(0), -1)
                negative_outputs = model(negative).last_hidden_state
                negative_outputs = F.adaptive_avg_pool2d(negative_outputs, (1, 1)).view(negative_outputs.size(0), -1)
        print('anchor outputs: ', anchor_outputs)
        print('positive outputs: ', positive_outputs)
        print('negative outputs: ', negative_outputs)

        loss = triplet_loss((anchor_outputs, positive_outputs, negative_outputs))
        
        if not val_flag:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_dict = {'train_loss': loss}
        else:
            loss_dict = {'val_loss': loss}

        if log_writer is not None:
            log_writer.update(loss_dict)
            if not val_flag:
                log_writer.set_step()
        
        return {'[Epoch] ' + k: loss for k, loss in loss_dict.items()}

def evaluate_topo(test_loader, model, model_config, positives_dict_path, device,
                testset, testset_size, batch_size, total_size, output_path,
                log_writer = None, embeddings_array = None, top_k = 1):
    model.eval()
    anchor_names = []
    all_embeddings = torch.zeros((total_size, 50 * model_config.hidden_size)).to(device)
    if embeddings_array is not None:
        for step, (anchor, anchor_name) in enumerate(test_loader):
            anchor_names.extend(anchor_name)
        all_embeddings = np.load(embeddings_array)
    else:
        for step, (anchor, anchor_name) in enumerate(test_loader):
            anchor_names.extend(anchor_name)
            with torch.no_grad():
                embedding = model(anchor.to(device)).last_hidden_state
                embedding = rearrange(embedding, 'b n d -> b (n d)')
                start_step = step * batch_size
                end_step = start_step + embedding.shape[0]
                indices = torch.tensor([x for x in range(start_step, end_step)]).long().to(device)
                all_embeddings[indices, :] = embedding
    
        #see whether closest match is correct
        all_embeddings = all_embeddings.cpu().detach()
    distances = euclidean_distances(all_embeddings, all_embeddings)
    correct = {}
    exact_pairs = {}
    incorrect = {}
    nonpaired = {}
    nonpaired_num = 0
    num_correct = 0
    with open(positives_dict_path, 'rb') as handle:
        positives_dict = pickle.load(handle)
    
    for index, entry in enumerate(distances):
        anchor_map_name = anchor_names[index]
        filename = anchor_map_name.split('/')[-1]
        if filename not in testset:
            continue
        
        distances[index][index] = float('inf')
        mins = np.argsort(distances[index])[0:top_k]
        closest_maps = [anchor_names[x] for x in mins]
        clean_anchor_name = filename.split('.')[0]
        clean_closest_maps = [x.split('/')[-1].split('.')[0] for x in closest_maps]
        if clean_anchor_name not in positives_dict.keys():
            nonpaired_num += 1
            nonpaired[clean_anchor_name] = clean_closest_maps
            continue
        correct_flag = False
        for clean_closest_map in clean_closest_maps:
            if clean_closest_map in positives_dict[clean_anchor_name]:
                num_correct += 1
                correct_flag = True
                exact_pairs[clean_anchor_name] = clean_closest_map
                correct[clean_anchor_name] = clean_closest_maps
        if not correct_flag:
            incorrect[clean_anchor_name] = clean_closest_maps
    print('CORRECT PAIRS: ', correct)
    print('INCORRECT PAIRS: ', incorrect)
        
    accuracy = num_correct / (testset_size-nonpaired_num) * 100
    print('Accuracy: ', accuracy)

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    correct_path = os.path.join(output_path, str(top_k) + 'correct.pkl')
    incorrect_path = os.path.join(output_path, str(top_k) + 'incorrect.pkl')
    nonpaired_path = os.path.join(output_path, str(top_k) + 'nonpaired.pkl')
    exact_pairs_path = os.path.join(output_path, str(top_k) + 'exact_pairs.pkl')
    np_path = os.path.join(output_path, 'embeddings.npy')

    with open(correct_path, 'wb') as f_corr:
        pickle.dump(correct, f_corr)
    with open(incorrect_path, 'wb') as f_incorr:
        pickle.dump(incorrect, f_incorr)
    with open(nonpaired_path, 'wb') as f_nonpair:
        pickle.dump(nonpaired, f_nonpair)
    with open(exact_pairs_path, 'wb') as f_exact_pair:
        pickle.dump(exact_pairs, f_exact_pair)

    if embeddings_array is None:
        np.save(np_path, all_embeddings)

def evaluate_geo_topo(topo_loader, geo_loader, model, model_config, positives_dict_path, device,
                batch_size, topo_size, geo_size, output_path,
                log_writer = None, topo_embeddings_array = None, geo_embeddings_array = None, top_k = 1,
                model_base = 'vit-mae'):
    model.eval()
    topo_anchor_names = []
    geo_anchor_names = []
    if model_base == 'vit-mae':
        embed_size = model_config.hidden_size
    else:
        embed_size = model_config.hidden_sizes[-1]
    topo_embeddings = torch.zeros((topo_size, embed_size)).to(device)
    geo_embeddings = torch.zeros((geo_size, embed_size)).to(device)
    if topo_embeddings_array is not None:
        print('Loading saved embeddings')
        for step, (anchor, anchor_name) in enumerate(topo_loader):
            topo_anchor_names.extend(anchor_name)
        topo_embeddings = np.load(topo_embeddings_array)
        print(topo_embeddings.shape)
        geo_embeddings = np.load(geo_embeddings_array)
        print(geo_embeddings.shape)
    else:
        for step, (anchor, anchor_name) in enumerate(topo_loader):
            topo_anchor_names.extend(anchor_name)
            with torch.no_grad():
                if model_base == 'vit-mae':
                    embedding = model(anchor.to(device)).last_hidden_state
                    embedding = embedding[:, -1, :] #get the last hidden state of the cls
                else:
                    output = model(anchor.to(device)).last_hidden_state
                    embedding = F.adaptive_avg_pool2d(output, (1, 1)).view(output.size(0), -1)
                    # embedding = rearrange(embedding, 'b n d -> b (n d)')
                    embedding = embedding
                start_step = step * batch_size
                end_step = start_step + embedding.shape[0]
                indices = torch.tensor([x for x in range(start_step, end_step)]).long().to(device)
                topo_embeddings[indices, :] = embedding

        for step, (anchor, anchor_name) in enumerate(geo_loader):
            geo_anchor_names.extend(anchor_name)
            with torch.no_grad():
                if model_base == 'vit-mae':
                    embedding = model(anchor.to(device)).last_hidden_state
                    embedding = embedding[:, -1, :] 
                else:
                    output = model(anchor.to(device)).last_hidden_state
                    embedding = F.adaptive_avg_pool2d(output, (1, 1)).view(output.size(0), -1)
                    embedding = embedding
                start_step = step * batch_size
                end_step = start_step + embedding.shape[0]
                indices = torch.tensor([x for x in range(start_step, end_step)]).long().to(device)
                geo_embeddings[indices, :] = embedding
        #see whether closest match is correct
        topo_embeddings = topo_embeddings.cpu().detach()
        geo_embeddings = geo_embeddings.cpu().detach()
    
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if topo_embeddings_array is None:
        topo_np_path = os.path.join(output_path, 'topo_embeddings.npy')
        np.save(topo_np_path, topo_embeddings)
    if geo_embeddings_array is None:
        geo_np_path = os.path.join(output_path, 'geo_embeddings.npy')
        np.save(geo_np_path, geo_embeddings)

    distances = euclidean_distances(geo_embeddings, topo_embeddings)
    correct = {}
    exact_pairs = {}
    correct_rank_dict = {}
    incorrect = {}
    nonpaired = {}
    nonpaired_num = 0
    num_correct = 0
    with open(positives_dict_path, 'rb') as handle:
        positives_dict = pickle.load(handle)
    
    #getting the root dir
    temp = topo_anchor_names[0].split('/')[-1]
    root_dir = topo_anchor_names[0].replace(temp, '')


    for index, entry in enumerate(distances):
        geo_anchor = geo_anchor_names[index]
        geo_filename = geo_anchor.split('/')[-1]
        
        mins = np.argsort(distances[index])[0:top_k]
        closest_maps = [topo_anchor_names[x] for x in mins]
        clean_anchor_name = geo_filename.split('.')[0]
        clean_closest_maps = [x.split('/')[-1].split('.')[0] for x in closest_maps]
        if clean_anchor_name not in positives_dict.keys():
            nonpaired_num += 1
            nonpaired[clean_anchor_name] = clean_closest_maps
            continue
        correct_flag = False
        for clean_closest_map in clean_closest_maps:
            if clean_closest_map in positives_dict[clean_anchor_name]:
                num_correct += 1
                correct_flag = True
                exact_pairs[clean_anchor_name] = clean_closest_map
                correct[clean_anchor_name] = clean_closest_maps
        #find rank of correct map
        min_paired_topo_rank = float('inf')
        for paired_topo in positives_dict[clean_anchor_name]:
            paired_topo_name = root_dir + paired_topo + '.tif'
            topo_idx = topo_anchor_names.index(paired_topo_name)
            rank = np.where(np.argsort(distances[index]) == topo_idx)
            rank = float(rank[0])
            if min_paired_topo_rank > rank:
                print('min_paired_topo_rank: ', min_paired_topo_rank)
                min_paired_topo_rank = rank
        correct_rank_dict[clean_anchor_name] = min_paired_topo_rank
        if not correct_flag:
            incorrect[clean_anchor_name] = clean_closest_maps
        
    accuracy = num_correct / (geo_size-nonpaired_num) * 100
    print('Accuracy: ', accuracy)

    correct_path = os.path.join(output_path, str(top_k) + 'correct.pkl')
    incorrect_path = os.path.join(output_path, str(top_k) + 'incorrect.pkl')
    nonpaired_path = os.path.join(output_path, str(top_k) + 'nonpaired.pkl')
    exact_pairs_path = os.path.join(output_path, str(top_k) + 'exact_pairs.pkl')
    corr_rank_path = os.path.join(output_path, str(top_k) + 'corr_rank.pkl')

    # with open(correct_path, 'wb') as f_corr:
    #     pickle.dump(correct, f_corr)
    # with open(incorrect_path, 'wb') as f_incorr:
    #     pickle.dump(incorrect, f_incorr)
    # with open(nonpaired_path, 'wb') as f_nonpair:
    #     pickle.dump(nonpaired, f_nonpair)
    # with open(exact_pairs_path, 'wb') as f_exact_pair:
    #     pickle.dump(exact_pairs, f_exact_pair)
    with open(corr_rank_path, 'wb') as corr_rank_pt:
        pickle.dump(correct_rank_dict, corr_rank_pt)

def split_train_val(dataset, train_pct=0.8, val_pct = 0.2, 
                            random_seed = 42, shuffle_dataset=True):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    val_split = int(np.floor(train_pct * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[:val_split], indices[val_split:]
    return train_indices, val_indices

if __name__ == '__main__':
    config_parser = parser = argparse.ArgumentParser(description='Training config')
    parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                        help='YAML config file specifying default arguments')

    parser = argparse.ArgumentParser(description='Topo ViT pre-training script')

    #Input data paths
    parser.add_argument('--topo_only', default= False, type=bool, help='Turn on if training topo to topo')
    parser.add_argument('--topo_train_input_path', type=str, help='topo train dataset path')
    parser.add_argument('--topo_full_input_path', type=str, help='full topomap dataset')
    parser.add_argument('--test_input_path', type=str, help='topo test input path')
    parser.add_argument('--positives_dict_path', type=str, help='path to dictionary of anchors + positives')
    parser.add_argument('--model_save_path', type=str, help='model save path')
    parser.add_argument('--results_save_path', type=str, help='save path for correct + incorrect pairs')
    parser.add_argument('--topo_distances_mtx_path', type=str, help='save path for embedding matrix')
    
    #Only need these if we are training geo-topo, not topo-topo
    parser.add_argument('--geo_distances_mtx_path', type=str, help='save path for embedding matrix')
    parser.add_argument('--geo_train_input_path', type=str, help='geologic map train dataset path')
    parser.add_argument('--geo_test_input_path', type=str, help='geologic map test dataset path')

    parser.add_argument('--model_type', default='resnet', type=str, help='resnet, c_vit-mae, vit-mae')
    parser.add_argument('--batch_size', default= 64, type=int, help='batch size')
    parser.add_argument('--learning_rate', default= 1e-6, type=int, help='learning rate')
    parser.add_argument('--start_epoch', default= 0, type=int, help='start epoch')
    parser.add_argument('--epochs', default= 100, type=int, help='end epoch')

    parser.add_argument('--top_k', default= 5, type=int, help='top-k for evaluation')

    parser.add_argument('--log_wandb', default= True, type=bool, help='Log training on Wandb')
    parser.add_argument('--wandb_project', type=str, help='wandb Project Name')
    parser.add_argument('--wandb_entity', type=str, help='wandb Entity Name')
    parser.add_argument('--wandb_run_name', type=str, help='wandb Run Name')
    parser.add_argument('--checkpoint_path', type=str, help='checkpoint storage path')
    parser.add_argument('--eval_only', action='store_true')  

    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)
    main(args)

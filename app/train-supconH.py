import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import torch
import time
import os
import pickle
from CLEAN.dataloader import *
from CLEAN.model import *
from CLEAN.utils import *
from CLEAN.losses import (
    SupConHardLoss,
    compute_embedding_compactness_stats,
    compute_gaussian_well_loss,
)
import torch.nn as nn
import argparse
from CLEAN.distance_map import get_dist_map


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--learning_rate', type=float, default=5e-4)
    parser.add_argument('-e', '--epoch', type=int, default=1500)
    parser.add_argument('-n', '--model_name', type=str, default='split10_supconH')
    parser.add_argument('-t', '--training_data', type=str, default='split10')
    # ------------  SupCon-Hard specific  ------------ #
    parser.add_argument('-T', '--temp', type=float, default=0.1)
    parser.add_argument('--n_pos', type=int, default=9)
    parser.add_argument('--n_neg', type=int, default=30)
    # ------------------------------------------- #
    parser.add_argument('-d', '--hidden_dim', type=int, default=512)
    parser.add_argument('-o', '--out_dim', type=int, default=256)
    parser.add_argument('--adaptive_rate', type=int, default=60)
    parser.add_argument('--verbose', type=bool, default=False)
    parser.add_argument('--use_gaussian_well', action='store_true')
    parser.add_argument('--lambda_well', type=float, default=0.05)
    parser.add_argument('--sigma_well', type=float, default=1.0)
    args = parser.parse_args()
    return args


def get_dataloader(dist_map, id_ec, ec_id, args):
    params = {
        'batch_size': 6000,
        'shuffle': True,
    }
    negative = mine_hard_negative(dist_map, 100)
    train_data = MultiPosNeg_dataset_with_mine_EC(
        id_ec, ec_id, negative, args.n_pos, args.n_neg, return_labels=True)
    train_loader = torch.utils.data.DataLoader(train_data, **params)
    return train_loader


def flatten_batch_embeddings(model_emb):
    return model_emb.reshape(-1, model_emb.size(-1))


def average_or_zero(total, count):
    if count == 0:
        return 0.0
    return total / count


def train(model, args, epoch, train_loader,
          optimizer, device, dtype, criterion):
    model.train()
    total_original_loss = 0.
    total_well_loss = 0.
    total_loss = 0.
    start_time = time.time()
    stats_totals = {
        'valid_well_sample_count': 0,
        'valid_well_class_count': 0,
        'intra_center_dist_sum': 0.0,
        'intra_center_dist_count': 0,
        'intra_pairwise_dist_sum': 0.0,
        'intra_pairwise_dist_count': 0,
        'nearest_negative_center_dist_sum': 0.0,
        'nearest_negative_center_dist_count': 0,
    }

    for batch, batch_data in enumerate(train_loader):
        optimizer.zero_grad()
        data, labels = batch_data
        model_emb = model(data.to(device=device, dtype=dtype))
        original_loss = criterion(model_emb, args.temp, args.n_pos)
        well_loss = model_emb.new_zeros(())
        well_stats = {
            'valid_sample_count': 0,
            'valid_class_count': 0,
        }

        flat_embeddings = flatten_batch_embeddings(model_emb)
        flat_labels = labels.to(device=device)
        compactness_stats = compute_embedding_compactness_stats(
            flat_embeddings, flat_labels.reshape(-1))

        if args.use_gaussian_well:
            well_loss, well_stats = compute_gaussian_well_loss(
                flat_embeddings, flat_labels.reshape(-1), args.sigma_well)

        loss = original_loss + args.lambda_well * well_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_original_loss += original_loss.item()
        total_well_loss += well_loss.item()
        stats_totals['valid_well_sample_count'] += well_stats['valid_sample_count']
        stats_totals['valid_well_class_count'] += well_stats['valid_class_count']
        stats_totals['intra_center_dist_sum'] += compactness_stats['intra_center_dist_sum']
        stats_totals['intra_center_dist_count'] += compactness_stats['intra_center_dist_count']
        stats_totals['intra_pairwise_dist_sum'] += compactness_stats['intra_pairwise_dist_sum']
        stats_totals['intra_pairwise_dist_count'] += compactness_stats['intra_pairwise_dist_count']
        stats_totals['nearest_negative_center_dist_sum'] += compactness_stats['nearest_negative_center_dist_sum']
        stats_totals['nearest_negative_center_dist_count'] += compactness_stats['nearest_negative_center_dist_count']

        if args.verbose:
            lr = args.learning_rate
            ms_per_batch = (time.time() - start_time) * 1000  
            if args.use_gaussian_well:
                print(f'| epoch {epoch:3d} | {batch:5d}/{len(train_loader):5d} batches | '
                      f'lr {lr:02.4f} | ms/batch {ms_per_batch:6.4f} | '
                      f'original_loss {original_loss.item():5.4f} | '
                      f'well_loss {well_loss.item():5.4f} | '
                      f'total_loss {loss.item():5.4f} | '
                      f'valid_well_sample_count {well_stats["valid_sample_count"]:4d} | '
                      f'valid_well_class_count {well_stats["valid_class_count"]:4d}')
            else:
                print(f'| epoch {epoch:3d} | {batch:5d}/{len(train_loader):5d} batches | '
                      f'lr {lr:02.4f} | ms/batch {ms_per_batch:6.4f} | '
                      f'loss {loss.item():5.2f}')
            start_time = time.time()
    # record running average training loss
    batch_count = batch + 1
    epoch_stats = {
        'original_loss': total_original_loss / batch_count,
        'well_loss': total_well_loss / batch_count,
        'total_loss': total_loss / batch_count,
        'valid_well_sample_count': stats_totals['valid_well_sample_count'],
        'valid_well_class_count': stats_totals['valid_well_class_count'],
        'avg_intra_center_dist': average_or_zero(
            stats_totals['intra_center_dist_sum'],
            stats_totals['intra_center_dist_count'],
        ),
        'avg_intra_pairwise_dist': average_or_zero(
            stats_totals['intra_pairwise_dist_sum'],
            stats_totals['intra_pairwise_dist_count'],
        ),
        'avg_nearest_negative_center_dist': average_or_zero(
            stats_totals['nearest_negative_center_dist_sum'],
            stats_totals['nearest_negative_center_dist_count'],
        ),
    }
    return epoch_stats

def main():
    seed_everything()
    ensure_dirs('./data/model')
    args = parse()
    if args.use_gaussian_well and args.sigma_well <= 0:
        raise ValueError('sigma_well must be > 0 when Gaussian well regularization is enabled.')
    torch.backends.cudnn.benchmark = True
    id_ec, ec_id_dict = get_ec_id_dict('./data/' + args.training_data + '.csv')
    ec_id = {key: list(ec_id_dict[key]) for key in ec_id_dict.keys()}
    #======================== override args ====================#
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    dtype = torch.float32
    lr, epochs = args.learning_rate, args.epoch
    model_name = args.model_name
    print('==> device used:', device, '| dtype used: ',
          dtype, "\n==> args:", args)
    if args.use_gaussian_well:
        print(f'==> Gaussian well regularization enabled: '
              f'lambda_well={args.lambda_well}, sigma_well={args.sigma_well}')
    else:
        print('==> Gaussian well regularization disabled')
    #======================== ESM embedding  ===================#
    # loading ESM embedding for dist map
 
    esm_emb = pickle.load(
        open('./data/distance_map/' + args.training_data + '_esm.pkl',
                'rb')).to(device=device, dtype=dtype)
    dist_map = pickle.load(open('./data/distance_map/' + \
        args.training_data + '.pkl', 'rb')) 
    #======================== initialize model =================#
    model = LayerNormNet(args.hidden_dim, args.out_dim, device, dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    criterion = SupConHardLoss
    best_loss = float('inf')
    train_loader = get_dataloader(dist_map, id_ec, ec_id, args)
    print("The number of unique EC numbers: ", len(dist_map.keys()))
    #======================== training =======-=================#
    # training
    for epoch in range(1, epochs + 1):
        if epoch % args.adaptive_rate == 0 and epoch != epochs + 1:
            optimizer = torch.optim.Adam(
                model.parameters(), lr=lr, betas=(0.9, 0.999))
            # save updated model
            torch.save(model.state_dict(), './data/model/' +
                       model_name + '_' + str(epoch) + '.pth')
            # delete last model checkpoint
            if epoch != args.adaptive_rate:
                os.remove('./data/model/' + model_name + '_' +
                          str(epoch-args.adaptive_rate) + '.pth')
            # sample new distance map
            dist_map = get_dist_map(
                ec_id_dict, esm_emb, device, dtype, model=model)
            train_loader = get_dataloader(dist_map, id_ec, ec_id, args)
        # -------------------------------------------------------------------- #
        epoch_start_time = time.time()
        epoch_stats = train(model, args, epoch, train_loader,
                            optimizer, device, dtype, criterion)
        train_loss = epoch_stats['total_loss']
        # only save the current best model near the end of training
        if (train_loss < best_loss and epoch > 0.8*epochs):
            torch.save(model.state_dict(), './data/model/' + model_name + '.pth')
            best_loss = train_loss
            print(f'Best from epoch : {epoch:3d}; loss: {train_loss:6.4f}')

        elapsed = time.time() - epoch_start_time
        print('-' * 75)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
              f'training loss {train_loss:6.4f}')
        print(f'| original_loss {epoch_stats["original_loss"]:6.4f} | '
              f'well_loss {epoch_stats["well_loss"]:6.4f} | '
              f'total_loss {epoch_stats["total_loss"]:6.4f}')
        print(f'| valid_well_sample_count {epoch_stats["valid_well_sample_count"]} | '
              f'valid_well_class_count {epoch_stats["valid_well_class_count"]}')
        print(f'| avg_intra_center_dist {epoch_stats["avg_intra_center_dist"]:6.4f} | '
              f'avg_intra_pairwise_dist {epoch_stats["avg_intra_pairwise_dist"]:6.4f} | '
              f'avg_nearest_negative_center_dist '
              f'{epoch_stats["avg_nearest_negative_center_dist"]:6.4f}')
        print('-' * 75)
    # remove tmp save weights when they were actually created
    best_model_path = './data/model/' + model_name + '.pth'
    periodic_ckpt_path = './data/model/' + model_name + '_' + str(epoch) + '.pth'
    if os.path.exists(best_model_path):
        os.remove(best_model_path)
    if os.path.exists(periodic_ckpt_path):
        os.remove(periodic_ckpt_path)
    # save final weights
    torch.save(model.state_dict(), best_model_path)


if __name__ == '__main__':
    main()

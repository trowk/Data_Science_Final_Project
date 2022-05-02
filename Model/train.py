from models import AutoEncoder, AutoEncoderConv, save_model
from utils import load_data, error, load_synthetic

import time
import torch
import torch.utils.tensorboard as tb
from tqdm import tqdm

TRAIN_PATH = "Data/new_train.npy"
VALID_PATH = "Data/new_test.npy"

def train(args):
    from os import path
    # model = AutoEncoder(encoder_dim_sizes = [32, 64, 32, 16], decoder_dim_sizes = [32, 64, 32], n_input = 20, latent_dim = 10)
    model = AutoEncoderConv(latent_dim=args.latent_dim)

    # Set hyperparameters from the parser
    lr = args.lr
    epochs = args.epoch
    batch_size = args.batchsize
    num_workers = args.num_workers
    weight_decay = args.weight_decay
    # Set up the cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Set up loss function and optimizer
    loss_func = torch.nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay = weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[30,60], gamma=0.1)

    # Set up training data and validation data
    if not args.synthetic:
        data_train = load_data(TRAIN_PATH, num_workers, batch_size, torch.nn.functional.normalize, store_norms=args.denormalize, noise=args.noise)
        data_val = load_data(VALID_PATH, num_workers, batch_size, torch.nn.functional.normalize, store_norms=args.denormalize, noise=None)
    else:
        data_train = load_synthetic(num_workers, batch_size, mean=0, std=1, normalize=True, data_len = 13504, store_data = False)
        data_val = load_data(VALID_PATH, num_workers, batch_size, torch.nn.functional.normalize, store_norms=args.denormalize, noise=None)

    experiment_data = list()

    # Wrap in a progress bar.
    for epoch in tqdm(range(epochs)):
        # Set the model to training mode.
        model.train()
        optim.zero_grad()

        train_error_val = list()
        loss_vals = list()
        for x in data_train:
            if args.denormalize:
                d, norms = x
                norms = norms.to(device)
            else:
                d = x
            d = d.to(device)
            pred = model(d)
            if args.denormalize:
                train_error_val.append(error(pred * norms[:, None], d * norms[:, None]))
            else:
                train_error_val.append(error(pred, d))
            # Compute loss and update model weights.
            loss = loss_func(pred, d)

            loss.backward()
            optim.step()
            optim.zero_grad()
            
            # Add loss to TensorBoard.
            loss_vals.append(loss.item())

        train_error_total = torch.FloatTensor(train_error_val).mean().item()
        loss_error_total = torch.FloatTensor(loss_vals).mean().item()
        scheduler.step()

        # Set the model to eval mode and compute accuracy.
        # No need to change this, but feel free to implement additional logging.
        model.eval()
        error_validation = list()
        val_loss = list()

        for x in data_val:
            if args.denormalize:
                d, norms = x
                norms = norms.to(device)
            else:
                d = x
            d = d.to(device)
            pred = model(d)
            
            if args.denormalize:
                error_validation.append(error(pred * norms[:, None], d * norms[:, None]))
            else:
                error_validation.append(error(pred, d))
                val_loss.append(loss_func(pred, d).item())

        error_total = torch.FloatTensor(error_validation).mean().item()
        experiment_data.append({'latent_dim': args.latent_dim, 'epoch': epoch, 'loss': loss_error_total, 'train_error': train_error_total, 'validation_error': error_total})
        print('Validation Loss:', torch.FloatTensor(val_loss).mean().item())


    # save_model(model, 'Testing/autoencoder.th')
    return experiment_data


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default=None)
    # Put custom arguments here
    parser.add_argument('-l', '--lr', type=float, default=1e-2)
    parser.add_argument('-e', '--epoch', type=int, default=10)
    parser.add_argument('-b', '--batchsize', type=int, default=512)
    parser.add_argument('-w', '--num_workers', type=int, default=16)
    parser.add_argument('-d', '--weight_decay', type=float, default=1e-6)
    parser.add_argument('-v', '--denormalize', default=False, action='store_true')
    parser.add_argument('-n', '--noise', default=None, type=float)
    parser.add_argument('-s', '--synthetic', default=False, action='store_true')
    parser.add_argument('-f', '--latent_dim', type=int, default=5)

    args = parser.parse_args()
    train(args)




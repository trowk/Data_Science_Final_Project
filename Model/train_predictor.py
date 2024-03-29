from models import Predictor, AutoEncoderConv, save_model, load_model
from utils import load_data, load_data_predict, error, load_synthetic

import time
import torch
import torch.utils.tensorboard as tb
from tqdm import tqdm

TRAIN_PATH = "Data/train.npy"
VALID_PATH = "Data/test.npy"

def train(args):
    from os import path
    # model = AutoEncoder(encoder_dim_sizes = [32, 64, 32, 16], decoder_dim_sizes = [32, 64, 32], n_input = 20, latent_dim = 10)
    model = load_model('Testing/autoencoder.th', AutoEncoderConv)
    predictor = Predictor() 

    # Set hyperparameters from the parser
    lr = args.lr
    epochs = args.epoch
    batch_size = args.batchsize
    num_workers = args.num_workers
    weight_decay = args.weight_decay
    # Set up the cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predictor = predictor.to(device)
    model = model.to(device)

    # Set up loss function and optimizer
    loss_func = torch.nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay = weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[30,60], gamma=0.1)

    # Set up training data and validation data
    data_train = load_data_predict(TRAIN_PATH, 'Data/param_train.npy', num_workers, batch_size, torch.nn.functional.normalize, store_norms=False)
    data_val = load_data_predict(VALID_PATH, 'Data/param_test.npy', num_workers, batch_size, torch.nn.functional.normalize, store_norms=False)

    experiment_data = list()

    # Wrap in a progress bar.
    for epoch in tqdm(range(epochs)):
        # Set the model to training mode.
        model.eval()
        predictor.train()
        optim.zero_grad()

        train_error_val = list()
        loss_vals = list()
        for x, params in data_train:
            x = x.to(device)
            params = params.to(device)
            pred = predictor(params)
            expanded = model.decoder(pred[:, None])

            train_error_val.append(error(expanded.squeeze(), x))
            # Compute loss and update model weights.
            loss = loss_func(expanded.squeeze(), x)

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
        predictor.eval()
        error_validation = list()
        val_loss = list()

        for x, params in data_val:
            x = x.to(device)
            params = params.to(device)
            pred = predictor(params)
            expanded = model.decoder(pred[:, None])

            error_validation.append(error(expanded.squeeze(), x))
            val_loss.append(loss_func(expanded.squeeze(), x).item())

        error_total = torch.FloatTensor(error_validation).mean().item()
        experiment_data.append({'epoch': epoch, 'loss': loss_error_total, 'train_error': train_error_total, 'validation_error': error_total})
        print('Validation Loss:', torch.FloatTensor(val_loss).mean().item())

    save_model(predictor, 'Testing/predictor.th')
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

    args = parser.parse_args()
    train(args)




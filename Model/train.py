from models import AutoEncoder, AutoEncoderConv, save_model
from utils import load_data, error

import time
import torch
import torch.utils.tensorboard as tb
from tqdm import tqdm

TRAIN_PATH = "Data/train.npy"
VALID_PATH = "Data/test.npy"

def train(args):
    from os import path
    # model = AutoEncoder(encoder_dim_sizes = [32, 64, 32, 16], decoder_dim_sizes = [32, 64, 32], n_input = 20, latent_dim = 10)
    model = AutoEncoderConv(encoder_dim_sizes = [32, 64, 32], decoder_dim_sizes = [32, 64, 32], n_input = 1, latent_dim = 10)

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

    # Set up training data and validation data
    data_train = load_data(TRAIN_PATH, num_workers, batch_size, torch.nn.functional.normalize)
    data_val = load_data(VALID_PATH, num_workers, batch_size, torch.nn.functional.normalize)

    # Set up loggers
    log_time = '{}'.format(time.strftime('%H-%M-%S'))
    log_name = 'lr=%s_epoch=%s_batch_size=%s_wd=%s' % (lr, epochs, batch_size, weight_decay)
    train_logger, valid_logger = None, None
    if args.log_dir:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train') + '/%s_%s' % (log_name, log_time))
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'test') + '/%s_%s' % (log_name, log_time))
    global_step = 0

    # Wrap in a progress bar.
    for epoch in tqdm(range(epochs)):
        # Set the model to training mode.
        model.train()

        train_error_val = list()
        loss_vals = list()
        for x in data_train:
            x = x.to(device)

            pred = model(x)
            
            train_error_val.append(error(pred, x))
            # Compute loss and update model weights.
            loss = loss_func(pred, x)

            loss.backward()
            optim.step()
            optim.zero_grad()
            
            # Add loss to TensorBoard.
            loss_vals.append(loss.item())
            if train_logger:
                train_logger.add_scalar('Loss', loss.item(), global_step=global_step)
            global_step += 1

        train_error_total = torch.FloatTensor(train_error_val).mean().item()
        print('Train Error', train_error_total)
        print('Loss', torch.FloatTensor(loss_vals).mean().item())
        if train_logger:
            train_logger.add_scalar('Train Error', train_error_total, global_step=global_step)

        # Set the model to eval mode and compute accuracy.
        # No need to change this, but feel free to implement additional logging.
        model.eval()

        error_validation = list()

        for x in data_val:
            x = x.to(device)
            pred = model(x)
            error_validation.append(error(pred, x))

        error_total = torch.FloatTensor(error_validation).mean().item()
        print('Validation Error', error_total)
        if valid_logger:
            valid_logger.add_scalar('Validation Error', error_total, global_step=global_step)

    save_model(model)


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




import sys
sys.path.append("../")
from models import AutoEncoderConv, Predictor, load_model
from utils import load_data_predict, error
import pickle
import torch
import numpy as np


def compress(args):
    model = load_model('autoencoder.th', AutoEncoderConv)
    predictor = load_model('predictor.th', Predictor)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    predictor = predictor.to(device)
    model.eval()
    predictor.eval()

    data = load_data_predict(args.dir, '../Data/param_train.npy', args.num_workers, args.batch_size, torch.nn.functional.normalize, store_norms=False, shuffle=False, drop_last=False)
    result = list()
    min_val = -1836615391510528.0
    max_val = 580880569466880.0
    for x, params in data:
        x = x.to(device)
        params = params.to(device)

        latent_pred = predictor(params)
        pred = model.decoder(latent_pred[:, None])
        result.append(pred.squeeze() * max_val + min_val)
    result = torch.cat(result)

    np.savetxt("predictions.csv", result.cpu().detach().numpy(), delimiter=",")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', type=str, default=None,
        help='Path to data to compress')
    parser.add_argument('-o', '--output', type=str, default="compressed.pickle",
        help='Path to data to compress')
        
    parser.add_argument('-b', '--batch_size', type=int, default=512)
    parser.add_argument('-w', '--num_workers', type=int, default=16)

    args = parser.parse_args()
    if not args.dir:
        print("Please input path to data")
        exit(-1)
    compress(args)
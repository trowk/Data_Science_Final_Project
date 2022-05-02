import sys
sys.path.append("../")
from models import AutoEncoderConv, load_model
from utils import load_data, error
import pickle
import torch


def compress(args):
    model = load_model('autoencoder.th', AutoEncoderConv)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    data = load_data(args.dir, args.num_workers, args.batch_size, torch.nn.functional.normalize, store_norms = True, shuffle=False, drop_last=False)
    result = list()
    for x, norms in data:
        x = x.to(device)
        norms = norms.to(device)
        compressed_data = model.encoder(x[:, None, :])
        result.append(torch.cat((compressed_data.squeeze()[:, None], norms[:, None]), 1))
    result = torch.cat(result)

    with open(args.output, 'wb') as f:
        pickle.dump(result, f)

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
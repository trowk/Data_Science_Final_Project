from models import AutoEncoderConv, load_model
from utils import load_data, error

def uncompress(args):
    model = load_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    with open(args.dir, 'rb') as f:
        data = pickle.load(f)

    for x, norm in data:
        x.to(device)
        pred = model.decoder(x)

        print(pred * norm[:, None])

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', type=str, default=None,
        help='Path to compressed data')
        
    parser.add_argument('-b', '--batchsize', type=int, default=1)
    parser.add_argument('-w', '--num_workers', type=int, default=16)

    args = parser.parse_args()
    if not args.dir:
        print("Plese input path to uncompressed data")
        exit(-1)
    uncompress(args)
from models import AutoEncoderConv, load_model
from utils import load_data, error
import pickle


def compress(args):
    model = load_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    data = load_data(args.dir, args.num_workers, args.batch_size, torch.nn.functional.normalize, store_norms = True)
    result = list()
    for x, norms in data:
        x.to(device)
        compressed_data = model.encoder(x)
        result.append((compressed_data, norms))

    with open(args.output, 'wb') as f:
        pickle.dump(result, f)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', type=str, default=None,
        help='Path to data to compress')
    parser.add_argument('-o', '--output', type=str, default="compressed.pickle",
        help='Path to data to compress')
        
    parser.add_argument('-b', '--batchsize', type=int, default=1)
    parser.add_argument('-w', '--num_workers', type=int, default=16)

    args = parser.parse_args()
    if not args.dir:
        print("Please input path to data")
        exit(-1)
    compress(args)
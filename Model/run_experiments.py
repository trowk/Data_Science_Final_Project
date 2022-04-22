from train import train

import csv

def run_experiments(args):
    csv_cols = ['Synthetic', 'Noise', 'epoch', 'loss', 'train_error', 'validation_error', 'Type']
    header = {'Synthetic':'Synthetic', 'Noise':'Noise', 'epoch':'epoch', 'loss':'loss', 'train_error':'train_error', 'validation_error': 'validation_error', 'Type': 'Type'}
    with open('Experiment/experiment.csv', newline='') as csv_file:
        reader = csv.DictReader(csv_file)
        with open("Experiment/results.csv", "w", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_cols)
            writer.writerow(header)
            for row in reader:
                args.noise = float(row['noise'])
                args.synthetic = row['synthetic_data'] == 'TRUE'
                result = train(args)
                result = [dict(item, **{'Type': int(row['type'])}) for item in result]
                for data in result:
                    writer.writerow(data)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default=None)
    # Put custom arguments here
    parser.add_argument('-l', '--lr', type=float, default=1e-2)
    parser.add_argument('-e', '--epoch', type=int, default=100)
    parser.add_argument('-b', '--batchsize', type=int, default=512)
    parser.add_argument('-w', '--num_workers', type=int, default=16)
    parser.add_argument('-d', '--weight_decay', type=float, default=1e-6)
    parser.add_argument('-v', '--denormalize', default=False, action='store_true')
    parser.add_argument('-n', '--noise', default=0, type=float)
    parser.add_argument('-s', '--synthetic', default=False, action='store_true')

    args = parser.parse_args()
    run_experiments(args)
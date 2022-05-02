from scipy.io import loadmat
from sklearn.model_selection import train_test_split

import argparse
import csv
import numpy as np
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train_file', type=str, default='S_train_unnormalized.csv',
        help='Path to training data')
    parser.add_argument('-d', '--data_store', type=str, default='../Model/Data',
        help='Directory where data will be stored')
    parser.add_argument('-s', '--seed', type=int, default=0,
        help='Random seed used to split the data')
    parser.add_argument('-v', '--validation_percentage', type=float, default=0.1,
        help='What percentage of the data that will belong to validation set')
    args = parser.parse_args()
    data = list()
    with open(args.train_file) as f:
        csvreader = csv.reader(f)
        for row in csvreader:
            data.append(row)
    
    data = np.array(data).astype(np.float64)  

    # train_file = os.path.join(args.data_store, 'predict_vals.npy')
    # with open(train_file, 'wb') as f:
    #     np.save(f, data)

    # data = list()
    # test = 0
    # with open('M.csv') as f:
    #     csvreader = csv.reader(f)
    #     for row in csvreader:
    #         data.append(row)
    
    # data = np.array(data).astype(np.float64)        
    # train_file = os.path.join(args.data_store, 'predict_params.npy')
    # with open(train_file, 'wb') as f:
    #     np.save(f, data)
    if args.validation_percentage > 0:
        x_train, x_test, _, _ = train_test_split([i for i in range(data.shape[0])], np.zeros(data.shape[0]), test_size=args.validation_percentage, random_state=args.seed)
    else:
        x_train = [i for i in range(data.shape[0])]
        x_test = []
    if not os.path.isdir(args.data_store):
        os.makedirs(args.data_store)
    train_file = os.path.join(args.data_store, 'train.npy')
    test_file = os.path.join(args.data_store, 'test.npy')
    with open(train_file, 'wb') as f:
        np.save(f, data[x_train])
    with open(test_file, 'wb') as f:
        np.save(f, data[x_test])

    data = list()
    with open('M.csv') as f:
        csvreader = csv.reader(f)
        for row in csvreader:
            data.append(row)
    
    data = np.array(data).astype(np.float64)        
    train_file = os.path.join(args.data_store, 'param_train.npy')
    test_file = os.path.join(args.data_store, 'param_test.npy')
    with open(train_file, 'wb') as f:
        np.save(f, data[x_train])
    with open(test_file, 'wb') as f:
        np.save(f, data[x_test])
    

    # x_train, x_test, _, _ = train_test_split(data, np.zeros(data.shape[0]), test_size=args.validation_percentage, random_state=args.seed)
    # if not os.path.isdir(args.data_store):
    #     os.makedirs(args.data_store)
    # train_file = os.path.join(args.data_store, 'large_train.npy')
    # test_file = os.path.join(args.data_store, 'large_test.npy')
    # with open(train_file, 'wb') as f:
    #     np.save(f, x_train)
    # with open(test_file, 'wb') as f:
    #     np.save(f, x_test)
    


if __name__ == "__main__":
    main()
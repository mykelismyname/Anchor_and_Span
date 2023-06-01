import numpy as np
import pickle
import pandas as pd
import json
import argparse

def read(pkl_file):
    with open(pkl_file, 'rb') as pkl:
        data = pickle.load(pkl)
        for i,entity in enumerate(data):
            print(i)


if __name__ == '__main__':
    par = argparse.ArgumentParser()
    par.add_argument('--data', default='../anns/annotations.json', help='location of the data')
    args = par.parse_args()
    read(args.data)

import os
import sys
import datetime
import argparse
import torch

from dataloader import ReceiptDataset, ReceiptDataLoader

import logging
logging.basicConfig(level=logging.INFO)




if "__main__" == __name__:
    arg_parser = argparse.ArgumentParser()
    
    arg_parser.add_argument('--run_name', type=str, default=None, help='name of the experiment')
    arg_parser.add_argument('--data', type=str, default=None, help='path to data folder')
    arg_parser.add_argument('--artefact_dir', type=str, default=None, help='path to output directory')
    
    arg_parser.add_argument('--ignored_classes', nargs='+', type=str, default=[], help='ignore annotation with this label')
    arg_parser.add_argument('--device', type=str, default='cuda:0', help='device (CPU, if CUDA not available)')
    arg_parser.add_argument('--use_large', default=False, action='store_true', help='use layoutlmv2-large-uncased as base model')
    arg_parser.add_argument('--save_all', action='store_true', help='')

    arg_parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    arg_parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    arg_parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    arg_parser.add_argument('--train_fraction', type=float, default=0.85, help='fraction of data to use for training (between 0 nad 1)')
    arg_parser.add_argument('--dropout', type=float, default=None, help='change dropout')
    arg_parser.add_argument('--clip_grad', type=float, default=None, help='gradient clipping value')
    arg_parser.add_argument('--early_stopping_patience', type=int, default=15, help='No of epochs to wait after min loss/max f1 score')
    arg_parser.add_argument('--stride', type=int, default=50, help='stride across tokens in case of overflow')
    arg_parser.add_argument('--max_length', type=int, default=512, help='max length of input sequence')
    arg_parser.add_argument('--seed', type=int, default=42, help='seed for reproducibility')

    args = arg_parser.parse_args()
    print(args._get_kwargs())

    if args.run_name is None:
        args.run_name = 'exp_layoulm'+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.data is None:
        raise ValueError('Data path must be specified')
    if args.artefact_dir is None:
        args.artefact_dir = os.makedirs(os.path.join('artefacts', args.run_name), exist_ok=True)
    else:
        args.artefact_dir = os.makedirs(os.path.join(args.artefact_dir, args.run_name), exist_ok=True)

    sys.stdout = open(os.path.join(args.artefact_dir, "layoutlm_log.log"), "a")
    sys.stderr = open(os.path.join(args.artefact_dir, "layoutlm_err.log"), "a")

    train_annotation, test_annotations =ReceiptDataLoader.get_train_test_split(args.data, args.train_fraction)
      
    
    

    
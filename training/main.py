import os
import sys
import json
import datetime
import traceback
import argparse
import tempfile
import mlflow

from dataloader import ReceiptDataLoader
from trainer import TrainCustomModel

import logging

logging.basicConfig(level=logging.INFO)


if "__main__" == __name__:
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument(
        "--run_name", type=str, default=None, help="name of the experiment"
    )
    arg_parser.add_argument(
        "--data", type=str, default=None, help="path to data folder"
    )
    arg_parser.add_argument("--bucket_name", type=str, default=None, help="bucket name")

    arg_parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="device (CPU, if CUDA not available)",
    )
    arg_parser.add_argument(
        "--use_large",
        action="store_true",
        help="use layoutlmv2-large-uncased as base model",
    )
    arg_parser.add_argument("--save_all", action="store_true", help="")

    arg_parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    arg_parser.add_argument(
        "--stride",
        type=int,
        default=50,
        help="stride across tokens in case of overflow",
    )
    arg_parser.add_argument(
        "--max_length", type=int, default=512, help="max length of input sequence"
    )
    arg_parser.add_argument(
        "--train_fraction",
        type=float,
        default=0.85,
        help="fraction of data to use for training (between 0 nad 1)",
    )
    arg_parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    arg_parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    arg_parser.add_argument(
        "--dropout", type=float, default=None, help="change dropout"
    )
    arg_parser.add_argument(
        "--clip_grad", type=float, default=None, help="gradient clipping value"
    )
    arg_parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=15,
        help="No of epochs to wait after min loss/max f1 score",
    )

    args = arg_parser.parse_args()
    logging.info(args._get_kwargs())

    if args.run_name is None:
        args.run_name = "exp_layoulm_" + datetime.datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S"
        )
    if args.data is None and args.bucket_name is None:
        raise ValueError("Either data or bucket_name should be provided")

    logging.info("Starting experiment")
    mlflow.set_experiment("exp_layoutlm")
    with mlflow.start_run(run_name=args.run_name) as run:
        mlflow.log_params(dict(args._get_kwargs()))

        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                sys.stdout = open(os.path.join(temp_dir, "layoutlm_out.log"), "a")
                sys.stderr = open(os.path.join(temp_dir, "layoutlm_err.log"), "a")

                logging.info("Params logged")

                dataloader = ReceiptDataLoader(
                    args.batch_size,
                    args.stride,
                    args.max_length,
                    args.train_fraction,
                    args.use_large,
                    args.data,
                    args.bucket_name,
                )

                open(os.path.join(temp_dir, "train_annotation.json"), "w").write(
                    json.dumps(dataloader.train_annotations)
                )
                open(os.path.join(temp_dir, "test_annotation.json"), "w").write(
                    json.dumps(dataloader.test_annotations)
                )

                logging.info("Data loaded")

                for idx, batch in enumerate(dataloader.train_dataloader):
                    if idx == 2:
                        break
                    logging.info(len(batch))
                logging.info(len(dataloader.train_dataloader))

                trainer = TrainCustomModel(
                    dataloader,
                    args.lr,
                    args.epochs,
                    args.dropout,
                    args.save_all,
                    args.clip_grad,
                    args.early_stopping_patience,
                )

                logging.info("Model loaded")

                # trainer.train()

                # logging.info("Training done")

                mlflow.log_artifact(os.path.join(temp_dir, "train_annotation.json"))
                mlflow.log_artifact(os.path.join(temp_dir, "test_annotation.json"))
            except Exception as e:
                mlflow.log_artifact(os.path.join(temp_dir, "layoutlm_out.log"))
                mlflow.log_artifact(os.path.join(temp_dir, "layoutlm_err.log"))
                raise e

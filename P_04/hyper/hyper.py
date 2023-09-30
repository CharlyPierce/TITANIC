
import tensorflow as tf

import google.cloud.aiplatform as aiplatform
from google.cloud.aiplatform import hyperparameter_tuning as hpt
from model import hyperparameter_tun
import argparse
import hypertune

def get_args():
  '''Parses args. Must include all hyperparameters you want to tune.'''

  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--dense1',
      required=True,
      type=int,
      help='dense1')
  parser.add_argument(
      '--dense2',
      required=True,
      type=int,
      help='dense2')
  parser.add_argument(
      '--lr',
      required=True,
      type=float,
      help='learning rate')
  parser.add_argument(
      '--epochs',
      required=True,
      type=int,
      help='epochs')
  parser.add_argument(
      '--batch',
      required=True,
      type=int,
      help='batch')
  args = parser.parse_args()
  return args

def main():
  args = get_args()
  hp_metric = hyperparameter_tun(d1=args.dense1,d2=args.dense2,lr=args.lr,epochs=args.epochs,batch=args.batch)
    
  hpt = hypertune.HyperTune()

  hpt.report_hyperparameter_tuning_metric(
      hyperparameter_metric_tag='accuracy',
      metric_value=hp_metric,
      global_step=args.epochs)


if __name__ == "__main__":
    main()

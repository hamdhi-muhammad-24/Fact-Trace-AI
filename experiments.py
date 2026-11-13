# experiments.py
'''CM3604 Deep Learning Coursework — Experimental Evaluation Suite'''
import os, sys, json, argparse

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baselines', action='store_true')
    parser.add_argument('--task1', action='store_true')
    parser.add_argument('--task2', action='store_true')
    parser.add_argument('--all', action='store_true')
    return parser.parse_args()

if __name__ == '__main__':
    args = _parse_args()
    print('Experiments suite initialized.')

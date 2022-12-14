import numpy as np
np.random.seed(0)
import torch
torch.autograd.set_detect_anomaly(True)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import argparse
import time

def parse_args():
    # input arguments
    parser = argparse.ArgumentParser(description='URAMN')

    parser.add_argument('--dataset', nargs='?', default='acm')

    parser.add_argument('--nb_epochs', type=int, default=1000)
    parser.add_argument('--hid_units', type=int, default=64)
    parser.add_argument('--lr', type = float, default = 0.001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--drop_prob', type=float, default=0.5)
    parser.add_argument('--reg_coef', type=float, default=0.01)
    parser.add_argument('--sup_coef', type=float, default=1.0)
    parser.add_argument('--alpha', type=float, default=0.1) # 0.1 for dblp, 0.1 for acm, 0.8 for yelp
    parser.add_argument('--beta', type=float, default=0.2)  # 0.2 for dblp, 0.2 for acm, 0.8 for yelp
    
    parser.add_argument('--gpu_num', type=int, default=1)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--nheads', type=int, default=8)
    parser.add_argument('--activation', nargs='?', default='relu')
    parser.add_argument('--isSemi', action='store_true', default=False)
    parser.add_argument('--isBias', action='store_true', default=False)
    parser.add_argument('--isAttn', action='store_true', default=False)

    return parser.parse_known_args()

def printConfig(args):
    args_names = []
    args_vals = []
    for arg in vars(args):
        args_names.append(arg)
        args_vals.append(getattr(args, arg))
    print(args_names)
    print(args_vals)

def main():
    args, unknown = parse_args()
    print("alpha: {}, beta: {}, weight_decay: {}, reg_coef: {}".format(args.alpha, args.beta, args.weight_decay, args.reg_coef))
    from models import URAMN
    embedder = URAMN(args)


    embedder.training()

if __name__ == '__main__':
    start = time.time()
    main()
    print("time: ",time.time()-start)

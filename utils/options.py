#!/usr/bin/env python

# -*- coding: utf-8 -*-

# Python version: 3.6



import argparse



def args_parser():

    parser = argparse.ArgumentParser()

    # federated arguments

    parser.add_argument('--epochs', type=int, default=50, help="rounds of training")

    parser.add_argument('--num_users', type=int, default=10, help="number of users: K")



    parser.add_argument('--local_ep', type=int, default=2, help="the number of local epochs: E")

    parser.add_argument('--local_bs', type=int, default=32, help="local batch size: B")

    parser.add_argument('--bs', type=int, default=32, help="test batch size")

    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")

    parser.add_argument('--lr_warmdown_rounds', type=int, default=0,

                        help='Linearly decay lr from --lr to --lr_warmdown_target over the first N global rounds; 0 disables.')

    parser.add_argument('--lr_warmdown_target', type=float, default=0.0,

                        help='Target lr for --lr_warmdown_rounds; keep this lr after warmdown. <=0 disables.')



    parser.add_argument('--num_edges', type=int, default=2, help="number of edge servers")

    parser.add_argument('--edge_rounds', type=int, default=2, help="L: edge aggregation rounds")

    parser.add_argument('--lambda_weight', type=float, default=1, help="global aggregation weight")

    parser.add_argument('--timesteps', default=20, type=int, help='simulation timesteps')

    parser.add_argument('--leak', default=0.995, type=float, help='membrane leak')

    parser.add_argument('--optimizer', default='SGD', type=str, help='optimizer for SNN backpropagation', choices=['SGD', 'Adam'])

    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay parameter for the optimizer')

    parser.add_argument('--momentum', type=float, default=0.9, help="SGD momentum (default: 0.5)")



    # model arguments

    parser.add_argument('--model', type=str, default='VGG9', help='model name')

    # other arguments

    parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['CIFAR10', 'DAGM2007'], help='dataset name')

    parser.add_argument('--dagm_data_dir', type=str, default='data/dagm2007',

                        help='Root directory for DAGM2007, containing Class*/Train and Class*/Test.')

    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')

    parser.add_argument('--dirichlet_alpha', type=float, default=0.5,

                        help='Dirichlet alpha for non-IID label partitioning when --iid is not set.')

    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")

    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")

    parser.add_argument('--verbose', action='store_true', help='verbose print')

    parser.add_argument('--seed', type=int, default=9, help='random seed (default: 1)')

    parser.add_argument('--eval_every', type=int, default=1, help='Frequency of model evaluation')

    parser.add_argument('--result_dir', type=str, default="results", help="Directory to store results")

    parser.add_argument('--snn', action='store_true', help='Compatibility flag; this public release trains SNN models only.')



    parser.add_argument('--parallel_workers', type=int, default='1', help="parallel_workers")

    parser.add_argument('--num_workers', type=int, default='0', help="num_workers")



    parser.add_argument('--mab_k', type=int, default=2, help="Number of clients to select per edge")

    parser.add_argument('--mab_delta', type=float, default=0.1, help="Exploration parameter")

    parser.add_argument('--mab_c', type=float, default=1.0, help="FOU constant")

    parser.add_argument('--mab_q', type=float, default=0.7, help="Reward weight")

    parser.add_argument('--client_selection', type=str, default='gossip',

                        choices=['mab', 'dmab', 'greedy', 'gossip', 'oort_loss', 'oort-loss'],

                        help='Client selection strategy: mab/dmab, greedy, gossip, or oort_loss.')

    parser.add_argument('--client_distance_profile', type=str, default='default',

                        choices=['default', 'extreme', 'loguniform'],

                        help='Client communication-distance profile.')

    parser.add_argument('--client_distance_min', type=float, default=30.0,

                        help='Minimum client distance in meters for non-default distance profiles.')

    parser.add_argument('--client_distance_max', type=float, default=800.0,

                        help='Maximum client distance in meters for non-default distance profiles.')

    parser.add_argument('--client_distance_seed', type=int, default=None,

                        help='Optional seed for client communication-distance profile only.')



    parser.add_argument('--dgc_enable', action='store_true', help='Enable DGC Scheme-A (compress model deltas)')

    parser.add_argument('--dgc_disable_ct', action='store_true',

                        help='Disable DGC-CT extra dense transmission while keeping ordinary DGC enabled.')

    parser.add_argument('--dgc_stat_enable', action='store_true',

                        help='Enable one-pass statistics of L2 norms for sent values and residuals (fallback DGC).')

    parser.add_argument('--dgc_stat_every', type=int, default=1,

                        help='Print & reset statistics every N rounds (default: 1).')

    parser.add_argument('--dgc_ratio', type=float, default=1, help='Target compression ratio (e.g., 600x)')

    parser.add_argument('--dgc_warmup', type=int, default=4, help='Warm-up epochs for sparsity')

    parser.add_argument('--dgc_fp16', action='store_true', help='Use FP16 for non-zero values when compressing')

    parser.add_argument('--res_decay', type=float, default=1,

                        help='Residual decay factor gamma (0 < gamma <= 1; smaller means faster forgetting).')

    parser.add_argument('--res_clip_norm', type=float, default=None,

                        help='Residual L2-norm clipping threshold tau; set None or 0 to disable.')

    parser.add_argument('--res_clip_warm', type=float, default=None,

                        help='Warmup-phase starting threshold for residual L2 clipping; linearly decays to res_clip_norm over warmup.')



    parser.add_argument('--eval_bs', type=int, default=64, help='eval batch size; 0 => use max(bs,64)')

    parser.add_argument('--eval_max_batches', type=int, default=0, help='only evaluate first N batches; 0=full')

    parser.add_argument('--eval_timesteps', type=int, default=0, help='override SNN timesteps during eval; 0=keep')

    parser.add_argument('--amp_eval', type=int, default=1, help='use AMP for eval on CUDA (1=True, 0=False)')

    args = parser.parse_args()

    return args




import argparse
import os


def args_parser():
    parser = argparse.ArgumentParser()
    path_dir = os.path.dirname(__file__)
    parser.add_argument('--path_cifar10', type=str, default=os.path.join(path_dir, 'data/CIFAR10/'))
    parser.add_argument('--path_cifar100', type=str, default=os.path.join(path_dir, 'data/CIFAR100/'))
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--num_clients', type=int, default=20)
    parser.add_argument('--num_online_clients', type=int, default=8)  #
    parser.add_argument('--num_rounds', type=int, default=200)

    parser.add_argument('--num_data_train', type=int, default=49000)

    parser.add_argument('--num_epochs_local_training', type=int, default=10)  #
    parser.add_argument('--batch_size_local_training', type=int, default=128)

    parser.add_argument('--total_steps', type=int, default=100)
    parser.add_argument('--server_steps', type=int, default=100)
    parser.add_argument('--mini_batch_size', type=int, default=20)
    parser.add_argument('--mini_batch_size_unlabeled', type=int, default=128)

    parser.add_argument('--batch_size_test', type=int, default=500)

    parser.add_argument('--lr_global_teaching', type=float, default=0.001)
    parser.add_argument('--lr_local_training', type=float, default=0.1)

    parser.add_argument('--temperature', type=float, default=2)
    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument('--ratio_imbalance', type=float, default=1.)
    parser.add_argument('--non_iid_alpha', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=7)

    parser.add_argument('--imb_type', default="exp", type=str, help='imbalance type')
    parser.add_argument('--imb_factor', default=0.01, type=float, help='imbalance factor')

    parser.add_argument('--rand_number', default=0, type=int, help='fix random number for data Dataset')

    parser.add_argument('--ensemble_ld', type=float, default=0.0)

    parser.add_argument('--ld', type=float, default=0.5)

# run:0.5

    # FedAvgM
    parser.add_argument('--init_belta', type=float, default=0.2)

    # FedProx
    parser.add_argument('--mu', type=float, default=0.0005)
    parser.add_argument('--gamma', type=float, default=0.1)

    args = parser.parse_args()

    return args

import argparse

def get_args():
    parser = argparse.ArgumentParser(description="model parameters")

    parser.add_argument('--hidden-size', type=int, default=512,
                        help='Internal hidden size of model.')

    parser.add_argument('--word-embed-dim', type=int, default=300,
                        help='Dimensions of word embeddings.')

    parser.add_argument('--resnet-out', type=int, default=2048,
                        help='Dimensions of Resnet features.')

    parser.add_argument('--mode', choices=['train', 'eval'])

    # Hyper-parameters
    parser.add_argument('--batch-size', type=int,
                        help="Batch size")
    parser.add_argument('--vbatch-size', type=int,
                        help="Batch size for validation")
    parser.add_argument('--val-split', type=float,
                        help="Validation set split proportion")
    parser.add_argument('--epoch', type=int,
                        help="Epochs to train")
    # Paths
    parser.add_argument('--data-root', type=str, default='data',
                        help="Location of data")
    parser.add_argument('--resume', type=str,
                        help="Location to resume model")
    parser.add_argument('--save', type=str,
                        help="Location to save model")

    # Others
    parser.add_argument('--cpu', action='store_true',
                        help="Set this to use CPU, default use CUDA")
    parser.add_argument('--n-workers', type=int, default=4,
                        help="How many processes for preprocessing")
    parser.add_argument('--pin-mem', type=bool, default=False,
                        help="DataLoader pin memory or not")
    parser.add_argument('--log-freq', type=int, default=100,
                        help="Logging frequency")
    parser.add_argument('--seed', type=int, default=420,
                        help="Random seed")

    return parser

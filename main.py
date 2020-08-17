import argparse


def def_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default='Breakout-ram-v4', help="name of the env")
    parser.add_argument("--alg_name", default='bdg', help="name of the algorithm")
    parser.add_argument("--seed", default=1, help="random seed")

    parser.add_argument("--buffer_size", default=1e5, help="size of the replay buffer")

    # algorithms details
    parser.add_argument("--return_k_gaussian", default=5, help="number of gaussians for return network")

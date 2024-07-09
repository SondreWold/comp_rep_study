import argparse

from generate import MarkovTree, generate_data
from operations import ALPHABET, BINARY_FUNC, FUNC_MAP, UNARY_FUNC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, help="The task to create data for")
    parser.add_argument(
        "--output_path", type=str, help="The path to save the data file to"
    )
    parser.add_argument(
        "--nr_samples", type=int, help="Number of samples to generate", default=100
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # These are the default settings from Hupkes et al (2020).
    alphabet_ratio = 1
    prob_func = 0.25
    lengths = [2, 3, 4, 5]
    placeholders = False
    omit_brackets = True
    random_probs = False

    if args.task in UNARY_FUNC:
        prob_unary = 1.0
        unary_functions = [FUNC_MAP[args.task]]
        binary_functions = []
    elif args.task in BINARY_FUNC:
        prob_unary = 0.0
        binary_functions = [FUNC_MAP[args.task]]
        unary_functions = []
    else:
        raise ValueError("Invalid task provided as argument")

    alphabet = [
        letter + str(i) for letter in ALPHABET for i in range(1, alphabet_ratio + 1)
    ]
    pcfg_tree_generator = MarkovTree(
        unary_functions=unary_functions,
        binary_functions=binary_functions,
        alphabet=alphabet,
        prob_unary=prob_unary,
        prob_func=prob_func,
        lengths=lengths,
        placeholders=placeholders,
        omit_brackets=omit_brackets,
    )

    output_file = generate_data(
        pcfg_tree=pcfg_tree_generator,
        total_samples=args.nr_samples,
        data_root=args.output_path,
        random_probs=random_probs,
    )

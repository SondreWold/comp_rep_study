ALPHABET = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
]


# UNARY
def copy(sequence):
    return sequence


def reverse(sequence):
    return sequence[::-1]


def shift(sequence):
    return sequence[1:] + [sequence[0]]


def echo(sequence):
    return sequence + [sequence[-1]]


def swap_first_last(sequence):
    return [sequence[-1]] + sequence[1:-1] + [sequence[0]]


def repeat(sequence):
    return sequence + sequence


# BINARY
def append(sequence1, sequence2):
    return sequence1 + sequence2


def prepend(sequence1, sequence2):
    return sequence2 + sequence1


def remove_first(sequence1, sequence2):
    return sequence2


def remove_second(sequence1, sequence2):
    return sequence1


UNARY_FUNC = ["copy", "reverse", "shift", "echo", "swap_first_last", "repeat"]
BINARY_FUNC = ["append", "prepend", "remove_first", "remove_second"]

FUNC_MAP = {
    "copy": copy,
    "reverse": reverse,
    "shift": shift,
    "echo": echo,
    "swap_first_last": swap_first_last,
    "repeat": repeat,
    "append": append,
    "prepend": prepend,
    "remove_first": remove_first,
    "remove_second": remove_second,
}

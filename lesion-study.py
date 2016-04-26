#!/usr/bin/env ipython

"""
Run lesion studies.

Usage:
    lesion_study.py [delete] [permute] [duplicate] [options]

Options:
    --num-ops=<n>   Number of operations to perform. Can be a string like "5-10" [default: 10]
"""

import json
import subprocess
import random
from docopt import docopt
opts = docopt(__doc__)

# all these ranges are inclusive, so you can swap layer 5 with layer 27 if you
# wish. (this is the resnet-200 model)
RANGES = [
        (5, 27),
        (29, 63),
]
DOWNSAMPLING_LAYERS = {1, 4, 28, 64}
MAX_LAYER = 66 # inclusive
PATH = "lesion-study-outs-200/large-scale/"

def number_or_range(n):
    n=str(n)
    if "-" in n:
        return random.randint(*map(int, n.split("-")))
    return int(n)

def run(delete=None, permute=None, duplicate=None, batchSize=8):
    result_file = "%s/%s.msgpack" % (
        PATH, "".join(["0123456789abcdef"[random.randint(0,15)] for _ in xrange(8)]))
    with open(result_file.replace(".msgpack", "-metadata.json"), "w") as f:
        json.dump({
            "result_file": result_file,
            "delete": list(delete or []),
            "permute": list(permute or []),
            "duplicate": list(duplicate or []),
            }, f)
    args = ([
        "th",
        "lesion-study.lua",
        "-batchSize", str(batchSize),
        "-data", "/mnt/net/data/",
        "-modelPath", "pretrained/resnet-200.t7",
        ]
        +(["-deleteBlock", ",".join([str(x) for x in delete])]
            if delete else [])
        +(["-permuteBlock", ",".join(["%s-%s" % row for row in permute])]
            if permute else [])
        +(["-duplicateBlock", ",".join([str(x) for x in duplicate])]
            if duplicate else [])
        +["-saveClassAccuracy", result_file]
    )
    print args
    subprocess.call(args)

def run_deletion_experiment():
    layers_to_delete = set([])
    nops = number_or_range(opts['--num-ops'])
    while len(layers_to_delete) < nops:
        layers_to_delete.add(random.randint(1, MAX_LAYER))
    run(delete = layers_to_delete)

def run_permutation_experiment():
    layers_to_permute = set([])
    nops = number_or_range(opts['--num-ops'])
    while len(layers_to_permute) < nops:
        # Can permute layers within certain ranges. Within each range,
        # we're guaranteed to not change the number of channels or
        # dimensionality of the input.
        range = random.choice(RANGES)
        a = random.randint(*range)
        b = random.randint(*range)
        if a != b:
            layers_to_permute.add( (a,b) )

    run(permute = layers_to_permute)

def run_duplication_experiment():
    layers_to_duplicate = set([])
    nops = number_or_range(opts['--num-ops'])
    while len(layers_to_duplicate) < nops:
        l = random.randint(1, MAX_LAYER)
        if l not in DOWNSAMPLING_LAYERS:
            layers_to_duplicate.add(l)
    run(duplicate = layers_to_duplicate, batchSize=4)

if __name__ == "__main__":
    operations = {
            "delete": run_deletion_experiment,
            "permute": run_permutation_experiment,
            "duplicate": run_duplication_experiment,
            }

    while True:
        operation = random.choice(operations.keys())
        if opts[operation]:
            operations[operation] ()

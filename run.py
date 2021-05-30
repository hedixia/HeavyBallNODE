from plane_vibration import *
from walker2d import *
import sys

run_pv = {
    'node': node_rnn_pv.main,
    'anode': anode_rnn_pv.main,
    'sonode': sonode_rnn_pv.main,
    'hbnode': hbnode_rnn_pv.main,
    'ghbnode': ghbnode_rnn_pv.main,
}

run_walker = {
    'node': node_rnn_walker.main,
    'anode': anode_rnn_walker.main,
    'sonode': sonode_rnn_walker.main,
    'hbnode': hbnode_rnn_walker.main,
    'ghbnode': ghbnode_rnn_walker.main,
}

all_models = {
    'pv': run_pv,
    'walker': run_walker,
}


def main(ds='pv', model='hbnode'):
    all_models[ds][model]()


if __name__ == '__main__':
    args = sys.argv[1:]
    assert len(args) == 2, "Input format: python3 run.py task model"
    print("Working on dataset {} using {} model".format(*args))
    main(*args)

import sys
sys.path.append('.')
import unittest
import matplotlib.pyplot as plt
from tree_plotter import (plot_node, decision_node, leaf_node, get_num_leafs, 
                          get_tree_depth)


def create_plot():
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    ax1 = plt.subplot(111, frameon=False)
    plot_node(ax1, 'a decision node', (0.5, 0.1), (0.1, 0.5), decision_node)
    plot_node(ax1, 'a leaf node', (0.8, 0.1), (0.3, 0.8), leaf_node)
    plt.show()


def _retreve_tree(i):
    list_of_trees = [{'no surfacing': {0: 'no', 
                                       1: {'flippers': {0: 'no', 1: 'yes'}}}},
                     {'no surfacing': {0: 'no', 
                                       1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 
                                                        1: 'no'}}}}]
    return list_of_trees[i]


class TestTreePlotter(unittest.TestCase):
    def setUp(self):
        self.tree0 = _retreve_tree(0)

    def test_get_num_leafs(self):
        self.assertEqual(get_num_leafs(self.tree0), 3,
                         'Wrong number of leafs')

    def test_get_tree_depth(self):
        self.assertEqual(get_tree_depth(self.tree0), 2,
                         'Wrong tree depth')

    def tearDown(self):
        del self.tree0


if __name__ == '__main__':
    unittest.main()
    create_plot()


import sys
sys.path.append('.')
import unittest
from tree import classify
from tree_plotter import retreve_tree


class TestTree(unittest.TestCase):
    def setUp(self):
        self.tree = retreve_tree(0)
        self.labels = ['no surfacing', 'flippers']

    def test_classify1(self):
        self.assertEqual(classify(self.tree, self.labels, [1, 0]), 'no',
                         'Wrong classification for sample [1, 0]')

    def test_classify2(self):
        self.assertEqual(classify(self.tree, self.labels, [1, 1]), 'yes',
                        'Wrong classification for sample [1, 1]')

    def tearDown(self):
        del self.tree
        del self.labels


if __name__ == "__main__":
    unittest.main()
import unittest

import numpy as np

from groupshift_model import AsymmetricSample


class AsymmetricSampleTests(unittest.TestCase):
    def test_returns_scope_in_group_id_order_for_group_one_node(self):
        np.random.seed(0)
        G = np.zeros((4, 2, 1, 2))
        N = np.array([[10.0], [20.0], [30.0], [40.0]])
        N_adj = np.array([[0, 0], [1, 1], [1, 1], [0, 0]])

        scope = AsymmetricSample(1, 1, "random").apply(
            G, N, N_adj, affected_nodes=np.array([1]), t=1
        )

        sampled_group_labels = [int(N_adj[members[0], 1]) for members in scope[0]]
        self.assertEqual(sampled_group_labels, [0, 1])

    def test_caps_sample_size_to_available_group_members(self):
        np.random.seed(0)
        G = np.zeros((4, 2, 1, 2))
        N = np.array([[10.0], [20.0], [30.0], [40.0]])
        N_adj = np.array([[0, 0], [1, 1], [1, 1], [1, 1]])

        scope = AsymmetricSample(2, 2, "random").apply(
            G, N, N_adj, affected_nodes=np.array([0]), t=1
        )

        self.assertEqual([len(members) for members in scope[0]], [1, 2])

    def test_returns_scope_for_every_group_in_multigroup_simulation(self):
        np.random.seed(0)
        G = np.zeros((6, 3, 1, 2))
        N = np.array([[10.0], [20.0], [30.0], [40.0], [50.0], [60.0]])
        N_adj = np.array([[0, 0], [0, 0], [1, 1], [1, 1], [2, 2], [2, 2]])

        scope = AsymmetricSample(1, 1, "random").apply(
            G, N, N_adj, affected_nodes=np.array([4]), t=1
        )

        sampled_group_labels = [int(N_adj[members[0], 1]) for members in scope[0]]
        self.assertEqual(sampled_group_labels, [0, 1, 2])


if __name__ == "__main__":
    unittest.main()

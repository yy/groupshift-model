import unittest

import numpy as np

from groupshift_model import AsymmetricSample, Attract, GroupshiftSim, Repulse


class OutputMetricsTests(unittest.TestCase):
    def test_rejects_multigroup_metrics_instead_of_silently_mislabeling_outgroups(self):
        sim = GroupshiftSim(
            G=np.zeros((6, 3, 1, 3)),
            N=np.array([[0.0], [10.0], [40.0], [50.0], [80.0], [90.0]]),
            N_adj=np.array(
                [
                    [0, 0, 0],
                    [0, 0, 0],
                    [1, 1, 1],
                    [1, 1, 1],
                    [2, 2, 2],
                    [2, 2, 2],
                ]
            ),
            C=np.array([]),
            num_nodes=6,
            dims=1,
            num_groups=3,
            timesteps=3,
            opinion_range=(0, 100),
            simnum=0,
            temp=1,
            init_method="initmeans",
            SCOPE=AsymmetricSample(1, 1, "random"),
            GLEAN=Attract(4.0, 0.2),
            SHIFT=Repulse(4.0, 1.0),
        )

        with self.assertRaisesRegex(
            ValueError,
            "outputMetrics currently requires exactly two groups",
        ):
            sim.outputMetrics(end_window=1)


if __name__ == "__main__":
    unittest.main()

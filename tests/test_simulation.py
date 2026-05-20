import unittest

import numpy as np

from groupshift_model import DoNothing, GroupshiftSim, IngroupOnly, Repulse


class SimulationTests(unittest.TestCase):
    def make_repulse_only_sim(self, affected_nodes):
        timesteps = len(affected_nodes)
        temp = len(affected_nodes[0])
        sim = GroupshiftSim(
            G=np.zeros((1, 2, 1, timesteps)),
            N=np.array([[10.0]]),
            N_adj=np.zeros((1, timesteps)),
            C=np.array(affected_nodes),
            num_nodes=1,
            dims=1,
            num_groups=2,
            timesteps=timesteps,
            opinion_range=(0, 100),
            simnum=0,
            temp=temp,
            init_method="initmeans",
            SCOPE=IngroupOnly(),
            GLEAN=DoNothing(),
            SHIFT=Repulse(10.0, 1.0),
        )
        sim.G[:, :, :, 0] = np.array([[[10.0], [20.0]]])
        sim.N_adj[:, 0] = 0
        return sim

    def test_duplicate_affected_nodes_are_applied_once_per_timestep(self):
        single = self.make_repulse_only_sim([[0], [0]])
        duplicated = self.make_repulse_only_sim([[0, 0], [0, 0]])

        single.run_simulation()
        duplicated.run_simulation()

        np.testing.assert_allclose(
            duplicated.G[0, :, 0, 1],
            single.G[0, :, 0, 1],
        )


if __name__ == "__main__":
    unittest.main()

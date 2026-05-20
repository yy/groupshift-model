import concurrent.futures
import os
import tempfile
import unittest

from groupshift_parallel import run_sims_to_csv


class SuccessfulSim:
    def initializeSim(self):
        pass

    def run_simulation(self):
        pass

    def outputMetrics(self):
        return [{"simnum": 1, "GroupID": 0}]

    def clearSim(self):
        pass


class FailingSim(SuccessfulSim):
    def run_simulation(self):
        raise RuntimeError("synthetic simulation failure")


class ParallelBatchTests(unittest.TestCase):
    def test_failed_simulation_aborts_without_publishing_partial_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "results.csv")

            with self.assertRaisesRegex(
                RuntimeError,
                "1 simulation\\(s\\) failed.*synthetic simulation failure",
            ):
                run_sims_to_csv(
                    [SuccessfulSim(), FailingSim()],
                    output_path,
                    max_workers=2,
                    executor_factory=concurrent.futures.ThreadPoolExecutor,
                    progress=lambda futures, total: futures,
                )

            self.assertFalse(os.path.exists(output_path))
            self.assertFalse(os.path.exists(f"{output_path}.tmp"))


if __name__ == "__main__":
    unittest.main()

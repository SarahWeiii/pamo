import numpy as np
import stage3
from stage3.utils import stage3_logger as logger


logger.setLevel("DEBUG")

np.random.seed(998244353)

config = stage3.config.Stage3Config()
config.max_particles = 2
config.energy_calcs = [stage3.energy.DummyEnergyCalculator]

system = stage3.system.Stage3System(config, "cuda")

n = config.max_particles
system.n_particles = n
x_np = np.random.randn(n, 3).astype(np.float32)

system.q.assign(x_np)

ec: stage3.energy.DummyEnergyCalculator = system._get_energy_calculator(
    stage3.energy.DummyEnergyCalculator
)
# print(f"A_np =\n{ec.A_np}")
# print(f"b_np =\n{ec.b_np}")
# print(f"c_np =\n{ec.c_np}")
# print(f"x_np =\n{x_np}")

system.step()
x_new_np = system.q.numpy()
system._compute_energy()
energy_new = system.energy.numpy()[0]

# print(f"energy_new =\n{energy_new}")

x_gt = np.linalg.solve(ec.A_np, -ec.b_np).reshape(-1, 3)

assert np.allclose(x_new_np, x_gt, atol=1e-6)

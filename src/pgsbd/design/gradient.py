"""256-start Sobol sequence sampling + Adam local optimization engine for Tier B Design."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from pgsbd.design.objective import (
    DesignEvaluator,
    DesignRequirements,
    unit_to_physical,
)


class GradientDesignEngine:
    """Gradient-based multi-start inverse design solver.

    Architecture:
    1. Quasi-random Sobol sampling: generates 256 well-distributed initial points in [0, 1]^5.
    2. Batch forward evaluation: evaluates all 256 initial points in parallel using vectorized PyTorch.
    3. Top-K selection: picks the 16 best candidate points with lowest loss.
    4. Parallel Adam optimization: runs 35 gradient descent steps on the top 16 candidates
       with learning rate 0.05 and cosine annealing.
    5. Deduplication & return: extracts physical parameters for top distinct optima.
    """

    def __init__(
        self,
        evaluator: DesignEvaluator | None = None,
        num_sobol_samples: int = 256,
        top_k: int = 16,
        num_adam_steps: int = 35,
        lr: float = 0.05,
    ):
        self.evaluator = evaluator or DesignEvaluator()
        self.num_sobol = num_sobol_samples
        self.top_k = top_k
        self.num_steps = num_adam_steps
        self.lr = lr

    def run_optimization(
        self,
        reqs: DesignRequirements,
        seed: int = 42,
    ) -> list[dict[str, Any]]:
        """Run 256-start Sobol + Adam optimization to synthesize optimal device parameters.

        Args:
            reqs: Design targets and tolerances.
            seed: Random seed for reproducible Sobol sequence scrambling.

        Returns:
            List of evaluated candidate design dictionaries ordered by loss.
        """
        torch.manual_seed(seed)
        np.random.seed(seed)

        # 1. Generate 256 Sobol points in [0, 1]^5
        sobol = torch.quasirandom.SobolEngine(dimension=5, scramble=True, seed=seed)
        u_init = sobol.draw(self.num_sobol).to(torch.float64)  # [256, 5]

        # If temperature is fixed, clamp dimension 1 to normalized 300K
        if reqs.fixed_temperature is not None:
            norm_t = (reqs.fixed_temperature - 300.0) / 200.0
            u_init[:, 1] = float(np.clip(norm_t, 0.0, 1.0))

        # 2. Vectorized initial evaluation
        with torch.no_grad():
            init_losses = self.evaluator.compute_loss_torch(u_init, reqs)  # [256]

        # 3. Select top K candidates
        top_indices = torch.topk(init_losses, k=self.top_k, largest=False).indices
        u_top = u_init[top_indices].clone().detach().requires_grad_(True)  # [top_k, 5]

        # 4. Adam optimization
        optimizer = torch.optim.Adam([u_top], lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_steps)

        for _ in range(self.num_steps):
            optimizer.zero_grad()
            # Constrain to [0, 1]^5 via projected gradient
            losses = self.evaluator.compute_loss_torch(u_top, reqs)
            total_loss = torch.sum(losses)
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                u_top.clamp_(0.0, 1.0)
                if reqs.fixed_temperature is not None:
                    norm_t = (reqs.fixed_temperature - 300.0) / 200.0
                    u_top[:, 1] = float(np.clip(norm_t, 0.0, 1.0))

        # 5. Extract evaluated candidates
        with torch.no_grad():
            final_losses = self.evaluator.compute_loss_torch(u_top, reqs).cpu().numpy()
            u_final = u_top.detach().cpu().numpy()

        sorted_order = np.argsort(final_losses)
        candidates = []
        seen_coords = []

        for idx in sorted_order:
            u_pt = u_final[idx]
            # Simple spatial deduplication (distance > 0.08 in unit space)
            is_dup = any(np.linalg.norm(u_pt - s) < 0.08 for s in seen_coords)
            if is_dup and len(candidates) >= 3:
                continue

            seen_coords.append(u_pt)
            phys = unit_to_physical(u_pt)
            cand = self.evaluator.evaluate_candidate(
                wf=float(phys["WF"]),
                temperature_k=float(phys["T"]),
                t_d_um=float(phys["t_D_um"]),
                n_net_cm3=float(phys["N_net_cm3"]),
                d_um=float(phys["d_um"]),
            )
            cand["optimization_loss"] = float(final_losses[idx])
            cand["normalized_u"] = u_pt.tolist()
            candidates.append(cand)

            if len(candidates) >= 8:
                break

        return candidates

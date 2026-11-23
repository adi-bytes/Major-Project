"""NSGA-II Multi-Objective Optimization Engine for Tier B Device Design."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize

from pgsbd.design.objective import (
    DesignEvaluator,
    DesignRequirements,
    unit_to_physical,
)


class SBDDesignProblem(Problem):
    """Pymoo Problem for multi-objective Schottky Barrier Diode synthesis.

    Objectives (minimization):
    1. f1: -V_br (maximize breakdown voltage)
    2. f2: R_on_sp (minimize specific on-resistance in mOhm*cm^2)
    3. f3: V_o (minimize turn-on voltage in V)
    4. f4: -I_on (maximize forward current at 4V)
    """

    def __init__(
        self,
        evaluator: DesignEvaluator,
        reqs: DesignRequirements,
    ):
        self.evaluator = evaluator
        self.reqs = reqs

        # Determine number of inequality constraints
        constrs = []
        if reqs.target_v_br is not None:
            constrs.append("v_br")
        if reqs.target_r_on_sp is not None:
            constrs.append("r_on_sp")
        if reqs.target_v_o is not None:
            constrs.append("v_o")
        if reqs.target_i_on_4v is not None:
            constrs.append("i_on")

        self.constrs = constrs
        super().__init__(
            n_var=5,
            n_obj=4,
            n_ieq_constr=len(constrs),
            xl=0.0,
            xu=1.0,
        )

    def _evaluate(self, x: np.ndarray, out: dict[str, Any], *args: Any, **kwargs: Any) -> None:
        u_torch = torch.tensor(x, dtype=torch.float64)

        if self.reqs.fixed_temperature is not None:
            norm_t = (self.reqs.fixed_temperature - 300.0) / 200.0
            u_torch[:, 1] = float(np.clip(norm_t, 0.0, 1.0))

        with torch.no_grad():
            p = unit_to_physical(u_torch)
            wf = p["WF"]
            t = p["T"]
            t_d = p["t_D_um"]
            n_net = p["N_net_cm3"]
            d_um = p["d_um"]

            # 1. Breakdown
            bd = self.evaluator.breakdown.calculate_torch(t_d, n_net)
            v_br = bd["v_br"].cpu().numpy()

            # 2. Forward metrics
            radius_cm = (d_um.cpu().numpy() / 2.0) * 1.0e-4
            anode_area_cm2 = np.pi * (radius_cm**2)

            i_curve = self.evaluator.core.forward(
                v_grid=self.evaluator.v_grid_torch,
                wf=wf,
                t_k=t,
                t_d_um=t_d,
                n_net_cm3=n_net,
                d_um=d_um,
                theta=self.evaluator.calib_theta,
            )
            from pgsbd.physics.extract_torch import extract_metrics_torch

            fom = extract_metrics_torch(
                v_grid=self.evaluator.v_grid_torch,
                currents=i_curve,
                area_cm2=1.711e-4,
            )
            r_diff = fom["r_diff"].cpu().numpy()
            ron_sp = r_diff * anode_area_cm2 * 1000.0
            v_o = fom["v_o"].cpu().numpy()
            i_on = i_curve[:, -1].cpu().numpy()

        # Objectives: [-V_br, Ron_sp, V_o, -I_on]
        f1 = -v_br
        f2 = ron_sp
        f3 = v_o
        f4 = -i_on
        out["F"] = np.column_stack([f1, f2, f3, f4])

        # Constraints (g <= 0)
        if len(self.constrs) > 0:
            g_list = []
            for c in self.constrs:
                if c == "v_br" and self.reqs.target_v_br is not None:
                    # v_br >= target => target - v_br <= 0
                    g_list.append(self.reqs.target_v_br - v_br)
                elif c == "r_on_sp" and self.reqs.target_r_on_sp is not None:
                    # ron_sp <= target => ron_sp - target <= 0
                    g_list.append(ron_sp - self.reqs.target_r_on_sp)
                elif c == "v_o" and self.reqs.target_v_o is not None:
                    g_list.append(v_o - self.reqs.target_v_o)
                elif c == "i_on" and self.reqs.target_i_on_4v is not None:
                    g_list.append(self.reqs.target_i_on_4v - i_on)
            out["G"] = np.column_stack(g_list)


class NSGADesignEngine:
    """NSGA-II multi-objective Pareto optimization engine for power device synthesis."""

    def __init__(
        self,
        evaluator: DesignEvaluator | None = None,
        pop_size: int = 40,
        n_gen: int = 25,
    ):
        self.evaluator = evaluator or DesignEvaluator()
        self.pop_size = pop_size
        self.n_gen = n_gen

    def run_optimization(
        self,
        reqs: DesignRequirements,
        seed: int = 42,
    ) -> list[dict[str, Any]]:
        """Run NSGA-II to compute non-dominated Pareto front designs."""
        problem = SBDDesignProblem(self.evaluator, reqs)
        algorithm = NSGA2(pop_size=self.pop_size)

        res = minimize(
            problem,
            algorithm,
            ("n_gen", self.n_gen),
            seed=seed,
            verbose=False,
        )

        candidates = []
        if res.X is not None and len(res.X) > 0:
            x_pts = np.atleast_2d(res.X).copy()
            if reqs.fixed_temperature is not None:
                # The temperature gene is neutral during search; force the requested value
                x_pts[:, 1] = float(np.clip((reqs.fixed_temperature - 300.0) / 200.0, 0.0, 1.0))
            for u_pt in x_pts:
                phys = unit_to_physical(u_pt)
                cand = self.evaluator.evaluate_candidate(
                    wf=float(phys["WF"]),
                    temperature_k=float(phys["T"]),
                    t_d_um=float(phys["t_D_um"]),
                    n_net_cm3=float(phys["N_net_cm3"]),
                    d_um=float(phys["d_um"]),
                )
                cand["normalized_u"] = u_pt.tolist()
                candidates.append(cand)

        return candidates

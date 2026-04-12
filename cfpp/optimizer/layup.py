"""Multi-start gradient descent layup optimizer.

Finds optimal fiber angles [a1, ..., an] for each shell layer
to minimize structural compliance (maximize stiffness).

Core formula (adjoint method, Pedersen 1989):
  dC/dai = -u^T * (dK/dai) * u

Papers:
  - Pedersen 1989: "On optimal orientation of orthotropic materials"
  - Nomura et al. 2019: "Orientation optimization using gradient descent"
  - Safonov 2019: "CFAO for polymer composite AM"
"""

import numpy as np
from skfem import *
from skfem.models.elasticity import linear_elasticity, lame_parameters


class LayupOptimizer:
    """Multi-start gradient descent optimizer for composite layup angles."""

    def __init__(self, mesh_path, E1=60e3, E2=3.5e3, G12=2e3, nu12=0.3,
                 n_layers=5, r_outer=25.0, r_inner=20.0):
        """
        Args:
            mesh_path: path to .msh file
            E1, E2, G12, nu12: orthotropic material properties (MPa)
            n_layers: number of shell layers
            r_outer, r_inner: shell radii (mm)
        """
        self.mesh_path = mesh_path
        self.E1, self.E2, self.G12, self.nu12 = E1, E2, G12, nu12
        self.n_layers = n_layers
        self.r_outer = r_outer
        self.r_inner = r_inner

        # Compute compliance matrix Q in material frame
        nu21 = nu12 * E2 / E1
        denom = 1 - nu12 * nu21
        self.Q = np.array([
            [E1 / denom, nu12 * E2 / denom, 0],
            [nu12 * E2 / denom, E2 / denom, 0],
            [0, 0, G12]
        ])

    def rotated_Q(self, alpha_rad):
        """Compute rotated stiffness matrix Q_bar(alpha)."""
        c = np.cos(alpha_rad)
        s = np.sin(alpha_rad)
        c2, s2, cs = c*c, s*s, c*s

        T = np.array([
            [c2, s2, 2*cs],
            [s2, c2, -2*cs],
            [-cs, cs, c2 - s2]
        ])
        T_inv = np.linalg.inv(T)
        return T_inv @ self.Q @ T_inv.T

    def dQ_dalpha(self, alpha_rad):
        """Analytical derivative of Q_bar with respect to alpha.

        dQ_bar/dalpha contains sin(2a), cos(2a), sin(4a), cos(4a) terms.
        """
        # Numerical derivative (simpler, sufficient for our purposes)
        eps = 1e-6
        Q_plus = self.rotated_Q(alpha_rad + eps)
        Q_minus = self.rotated_Q(alpha_rad - eps)
        return (Q_plus - Q_minus) / (2 * eps)

    def effective_E(self, alpha_rad):
        """Compute effective AXIAL (Z-direction) Young's modulus for +-alpha layup.

        For bending of a cylinder, what matters is E_zz (axial modulus),
        NOT E along the fiber direction.
          alpha=0 (hoop fiber) -> E_zz = E2 (transverse, LOW)
          alpha=90 (axial fiber) -> E_zz = E1 (fiber direction, HIGH)

        Uses rotated compliance: E_zz = 1/S_zz
        """
        # Average Q for +-alpha
        Q_plus = self.rotated_Q(alpha_rad)
        Q_minus = self.rotated_Q(-alpha_rad)
        Q_avg = (Q_plus + Q_minus) / 2.0

        # Compliance matrix S = inv(Q)
        S_avg = np.linalg.inv(Q_avg)

        # In our convention: index 0=x(circumferential), 1=y(axial/Z), 2=shear
        # E_zz = 1 / S[1,1]
        E_zz = 1.0 / S_avg[1, 1]
        return E_zz

    def compute_compliance(self, alphas_deg, traction=(0.7074, 0, 0)):
        """Compute structural compliance for given layup angles.

        Uses volume-averaged effective E for each shell layer.

        Args:
            alphas_deg: [a1, ..., an] in degrees
            traction: applied traction (MPa)

        Returns:
            compliance: scalar
            displacement_max: scalar
        """
        import meshio
        from skfem.io.meshio import from_meshio

        # Average E across all layers weighted by shell volume
        shell_radii = np.linspace(self.r_outer, self.r_inner +
                                   (self.r_outer-self.r_inner)/self.n_layers,
                                   self.n_layers)

        E_avg = 0
        total_vol = 0
        for i, r in enumerate(shell_radii):
            alpha_rad = np.radians(alphas_deg[i])
            # For +/-alpha layup, average the +alpha and -alpha contributions
            E_plus = self.effective_E(alpha_rad)
            E_minus = self.effective_E(-alpha_rad)
            E_layer = (E_plus + E_minus) / 2

            # Shell volume proportional to 2*pi*r
            vol = 2 * np.pi * r
            E_avg += E_layer * vol
            total_vol += vol

        E_avg /= total_vol

        # Solve FEA with effective E
        mio = meshio.read(self.mesh_path)
        mesh = from_meshio(mio)
        e = ElementVector(ElementTetP1(), 3)
        ib = Basis(mesh, e)

        lam, mu = lame_parameters(float(E_avg), float(self.nu12))
        K = linear_elasticity(lam, mu).assemble(ib)

        traction = np.array(traction, dtype=float)

        # Load boundary (top face)
        load_facets = mesh.boundaries.get('load',
            mesh.facets_satisfying(lambda x: np.abs(x[2] - np.max(mesh.p[2])) < 1e-6))
        fb = FacetBasis(mesh, e, facets=load_facets)

        @LinearForm
        def surface_load(v, w):
            return traction[0]*v.value[0] + traction[1]*v.value[1] + traction[2]*v.value[2]

        f = surface_load.assemble(fb)

        fixed_facets = mesh.boundaries.get('fixed',
            mesh.facets_satisfying(lambda x: np.abs(x[0]) < 1e-6))
        if 'fixed' not in mesh.boundaries:
            fixed_facets = mesh.facets_satisfying(lambda x: np.abs(x[2]) < 1e-6)

        fixed_dofs = ib.get_dofs(fixed_facets).all()
        u = solve(*condense(K, f, D=fixed_dofs))

        compliance = float(f.dot(u))
        disp_max = float(np.max(np.abs(u)))

        return compliance, disp_max

    def compute_sensitivity(self, alphas_deg, traction=(0.7074, 0, 0)):
        """Compute dC/da for each layer using finite differences.

        For production use, implement analytical sensitivity.
        For our purposes, central finite difference is sufficient and simpler.
        """
        n = len(alphas_deg)
        dC_dalpha = np.zeros(n)
        delta = 0.5  # degrees

        C0, _ = self.compute_compliance(alphas_deg, traction)

        for i in range(n):
            alphas_plus = alphas_deg.copy()
            alphas_plus[i] += delta
            C_plus, _ = self.compute_compliance(alphas_plus, traction)

            alphas_minus = alphas_deg.copy()
            alphas_minus[i] -= delta
            C_minus, _ = self.compute_compliance(alphas_minus, traction)

            dC_dalpha[i] = (C_plus - C_minus) / (2 * delta)

        return dC_dalpha, C0

    def optimize(self, traction=(0.7074, 0, 0),
                 n_starts=8, max_iter=20,
                 alpha_min=20, alpha_max=85,
                 delta_max=5.0, tol=0.001):
        """Run multi-start gradient descent optimization.

        Args:
            traction: applied traction
            n_starts: number of random starting points
            max_iter: max gradient descent iterations per start
            alpha_min, alpha_max: angle bounds (degrees)
            delta_max: max angle change per iteration (degrees)
            tol: convergence tolerance (relative change in C)

        Returns:
            result: dict with optimal angles, compliance history, all starts
        """
        # Generate starting points
        starts = [
            np.full(self.n_layers, 30.0),              # all low
            np.full(self.n_layers, 45.0),              # all +/-45
            np.full(self.n_layers, 60.0),              # all mid
            np.full(self.n_layers, 80.0),              # all high
            np.linspace(30, 80, self.n_layers),        # gradient
            np.array([80,60,45,60,80])[:self.n_layers], # symmetric bending
            np.array([45,45,50,80,80])[:self.n_layers], # current design
        ]
        # Add random starts
        while len(starts) < n_starts:
            starts.append(np.random.uniform(alpha_min, alpha_max, self.n_layers))
        starts = starts[:n_starts]

        all_results = []

        for si, alpha0 in enumerate(starts):
            alpha = alpha0.copy()
            history = []

            print(f"  Start {si+1}/{n_starts}: alpha0={[f'{a:.0f}' for a in alpha]}")

            for it in range(max_iter):
                # Compute sensitivity
                dC, C = self.compute_sensitivity(alpha, traction)
                history.append({'iter': it, 'C': C, 'alpha': alpha.copy()})

                # Step size (Nomura NGM: normalize by max gradient)
                max_grad = np.max(np.abs(dC))
                if max_grad < 1e-12:
                    break
                eta = delta_max / max_grad

                # Gradient step
                alpha_new = alpha - eta * dC

                # Project to bounds
                alpha_new = np.clip(alpha_new, alpha_min, alpha_max)

                # Convergence check
                if it > 0 and abs(C - history[-2]['C']) / abs(history[-2]['C']) < tol:
                    print(f"    Converged at iter {it}: C={C:.6f}")
                    break

                alpha = alpha_new

            # Final compliance
            C_final, disp_final = self.compute_compliance(alpha, traction)

            all_results.append({
                'start_idx': si,
                'alpha_init': alpha0.tolist(),
                'alpha_opt': alpha.tolist(),
                'C_final': C_final,
                'disp_max': disp_final,
                'history': history,
                'n_iter': len(history),
            })

            print(f"    Final: alpha={[f'{a:.1f}' for a in alpha]}, C={C_final:.6f}, d={disp_final:.6f}")

        # Find best
        best_idx = np.argmin([r['C_final'] for r in all_results])
        best = all_results[best_idx]

        # Compute improvement vs current design
        current = np.array([45, 45, 50, 80, 80])[:self.n_layers]
        C_current, d_current = self.compute_compliance(current, traction)
        improvement = (C_current - best['C_final']) / C_current * 100

        print(f"\n=== Optimization Result ===")
        print(f"  Best start: {best_idx} ({best['n_iter']} iterations)")
        print(f"  Optimal: [{', '.join(f'+/-{a:.0f} deg' for a in best['alpha_opt'])}]")
        print(f"  C_opt={best['C_final']:.6f} vs C_current={C_current:.6f}")
        print(f"  Improvement: {improvement:.1f}%")
        print(f"  d_max: {best['disp_max']:.6f} mm")

        return {
            'best': best,
            'all_results': all_results,
            'C_current': C_current,
            'improvement_pct': improvement,
            'current_layup': current.tolist(),
        }

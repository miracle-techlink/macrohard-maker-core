"""SIMP 拓扑优化器 (Solid Isotropic Material with Penalization)

确定最优材料密度分布 (0=不放纤维, 1=放纤维),
用于指导路径规划的纤维布局区域.

SIMP 公式:  E(x) = E_base + x^p * (E_fiber - E_base)

参考:
  - Bendsoe & Sigmund, "Topology Optimization", Springer, 2003
  - Sigmund, "A 99 line topology optimization code", SMO 21(2), 2001
"""

import numpy as np
from numpy.typing import NDArray
from pathlib import Path

from skfem import *
from skfem.models.elasticity import lame_parameters
from skfem.helpers import ddot, sym_grad
from skfem.io.meshio import from_meshio
import meshio
from scipy.spatial import cKDTree


class SIMPOptimizer:
    """SIMP 拓扑优化器

    确定最优材料密度分布 (0=不放纤维, 1=放纤维)
    用于指导路径规划的纤维布局区域
    """

    def __init__(
        self,
        mesh_path: str,
        volume_fraction: float = 0.4,
        penalty: float = 3.0,
    ):
        """
        Args:
            mesh_path: 网格文件路径 (.msh)
            volume_fraction: 目标体积分数 (0.4 = 40% 的区域放纤维)
            penalty: SIMP 惩罚因子 (通常 3)
        """
        self.mesh_path = Path(mesh_path)
        self.volume_fraction = volume_fraction
        self.penalty = penalty

        # 加载网格
        mio = meshio.read(str(self.mesh_path))
        self.mesh_sk = from_meshio(mio)
        m = self.mesh_sk
        n_nodes = m.p.shape[1]
        n_elems = m.t.shape[1]
        print(f"[SIMP] Mesh: {n_nodes} nodes, {n_elems} elements")

        # 单元中心坐标
        self.centroids = m.p[:, m.t].mean(axis=1).T  # (n_elems, 3)
        self.n_elems = n_elems

        # 结果 (optimize 后填充)
        self.densities = None
        self.compliance_history = []

    def optimize(
        self,
        E_fiber: float = 60e3,
        E_base: float = 3.5e3,
        nu: float = 0.3,
        traction: tuple = (0, 0, -10),
        fixed_boundary: str = "fixed",
        load_boundary: str = "load",
        max_iter: int = 50,
        tol: float = 0.01,
        filter_radius: float = 3.0,
    ) -> dict:
        """运行 SIMP 优化

        SIMP 公式: E(x) = E_base + x^p * (E_fiber - E_base)

        Algorithm:
          1. Initialize density x = volume_fraction for all elements
          2. For each iteration:
             a. Compute effective E per element
             b. Assemble & solve FEA with spatially varying material
             c. Compute element strain energies (compliance sensitivity)
             d. Apply density filter (sensitivity averaging)
             e. Update densities using optimality criteria (bisection)
             f. Check convergence

        Returns:
            dict with:
                densities: (N,) element densities 0-1
                compliance_history: list of compliance values
                centroids: (N, 3) element centers
                fiber_region: (N,) bool mask where density > 0.5
                n_iterations: int
        """
        m = self.mesh_sk
        p = self.penalty
        vf = self.volume_fraction
        n_elems = self.n_elems
        traction = np.array(traction, dtype=float)

        # Element and basis
        e = ElementVector(ElementTetP1(), 3)
        ib = Basis(m, e)

        # Number of quadrature points per element
        # dx shape: (n_elems, n_qp)
        n_qp = ib.dx.shape[1] if ib.dx.ndim == 2 else 1

        # ---------- Boundary setup ----------
        if load_boundary in m.boundaries:
            load_facets = m.boundaries[load_boundary]
        else:
            load_facets = m.facets_satisfying(
                lambda x: np.abs(x[2] - np.max(m.p[2])) < 1e-6
            )

        if fixed_boundary in m.boundaries:
            fixed_facets = m.boundaries[fixed_boundary]
        else:
            fixed_facets = m.facets_satisfying(
                lambda x: np.abs(x[2]) < 1e-6
            )

        fb = FacetBasis(m, e, facets=load_facets)
        fixed_dofs = ib.get_dofs(fixed_facets).all()

        # ---------- Load vector (constant across iterations) ----------
        @LinearForm
        def surface_load(v, w):
            return (traction[0] * v.value[0]
                    + traction[1] * v.value[1]
                    + traction[2] * v.value[2])

        f_vec = surface_load.assemble(fb)

        # ---------- Compute element volumes ----------
        # Use scalar basis for element volume integration
        e_scalar = ElementTetP1()
        ib_scalar = Basis(m, e_scalar)
        # dx shape: (n_elems, n_qp) — sum over quadrature points for each element
        elem_volumes = ib_scalar.dx.sum(axis=1) if ib_scalar.dx.ndim == 2 \
            else ib_scalar.dx

        # ---------- Build unit stiffness (E=1, nu=nu) ----------
        # We'll scale element contributions by E_eff(x) each iteration.
        # For efficiency: assemble K0 once with E=1, then scale per-element.
        lam0, mu0 = lame_parameters(1.0, nu)

        # We need per-element stiffness scaling, so we use a custom bilinear
        # form that accepts element-wise Young's modulus.
        def _assemble_scaled_K(E_elem):
            """Assemble global stiffness with element-wise E.

            Uses Lame parameters scaled by E: lam = E * lam0, mu = E * mu0
            where lam0, mu0 are for E=1.
            """
            # Map element E to quadrature points
            # scikit-fem expects (n_elems, n_qp) for element-wise data
            E_qp = np.broadcast_to(E_elem[:, np.newaxis], (n_elems, n_qp))

            lam_qp = lam0 * E_qp
            mu_qp = mu0 * E_qp

            @BilinearForm
            def variable_elasticity(u, v, w):
                lam = w.lam
                mu = w.mu

                # Symmetric gradient (strain tensor)
                def eps(field):
                    return 0.5 * (field.grad + field.grad.transpose((1, 0, 2, 3)))

                def sigma(T):
                    # 3D isotropic: sigma = 2*mu*eps + lam*tr(eps)*I
                    tr = T[0, 0] + T[1, 1] + T[2, 2]
                    eye = np.zeros_like(T)
                    eye[0, 0] = 1.0
                    eye[1, 1] = 1.0
                    eye[2, 2] = 1.0
                    return 2.0 * mu * T + lam * tr * eye

                return ddot(sigma(eps(u)), eps(v))

            K = variable_elasticity.assemble(ib, lam=lam_qp, mu=mu_qp)
            return K

        # ---------- Compute element strain energy from displacement ----------
        def _element_strain_energy(u_vec, E_elem):
            """Compute strain energy per element: ce_e = u_e^T K_e u_e

            We compute this via integrating sigma:epsilon over each element.
            """
            du = ib.interpolate(u_vec)
            grad_u = du.grad  # (3, 3, n_elems, n_qp)

            if grad_u.ndim == 3:
                grad_u = grad_u[:, :, :, np.newaxis]  # (3,3,n_elems,1)

            # Strain: eps = 0.5 * (grad_u + grad_u^T)
            eps = 0.5 * (grad_u + grad_u.transpose((1, 0, 2, 3)))
            # eps shape: (3, 3, n_elems, n_qp)

            n_qp_actual = eps.shape[3]
            # E_qp shape: (n_elems, n_qp)
            E_qp = np.broadcast_to(E_elem[:, np.newaxis], (n_elems, n_qp_actual))

            lam_local = lam0 * E_qp  # (n_elems, n_qp)
            mu_local = mu0 * E_qp

            tr_eps = eps[0, 0] + eps[1, 1] + eps[2, 2]  # (n_elems, n_qp)

            # strain_energy_density = 2*mu*(eps:eps) + lam*tr(eps)^2
            eps_sq = np.sum(eps * eps, axis=(0, 1))  # (n_elems, n_qp)
            sed = 2.0 * mu_local * eps_sq + lam_local * tr_eps ** 2

            # Integrate over element: dx shape (n_elems, n_qp)
            ce = np.sum(sed * ib.dx, axis=1)  # (n_elems,)
            return ce

        # ---------- Density filter ----------
        # Build neighbor structure once
        tree = cKDTree(self.centroids)
        # Precompute neighbor lists and weights for filter
        filter_neighbors = []
        filter_weights = []
        for i in range(n_elems):
            nbrs = tree.query_ball_point(self.centroids[i], filter_radius)
            dists = np.linalg.norm(
                self.centroids[nbrs] - self.centroids[i], axis=1
            )
            w = np.maximum(filter_radius - dists, 0.0)
            filter_neighbors.append(np.array(nbrs))
            filter_weights.append(w)

        def _density_filter(dc, x):
            """Sensitivity filter (weighted average within radius)."""
            filtered = np.zeros_like(dc)
            for i in range(n_elems):
                nbrs = filter_neighbors[i]
                w = filter_weights[i]
                denom = np.sum(w * x[nbrs])
                if denom > 1e-12:
                    filtered[i] = np.sum(w * x[nbrs] * dc[nbrs]) / denom
                else:
                    filtered[i] = dc[i]
            return filtered

        # ---------- Optimality criteria update ----------
        def _oc_update(x, dc, vol_target):
            """Bisection-based optimality criteria update."""
            x_new = np.copy(x)
            x_min = 1e-3  # minimum density to avoid singularity
            x_max = 1.0
            move = 0.2  # max density change per iteration

            l1, l2 = 0.0, 1e9
            while (l2 - l1) / (l1 + l2 + 1e-12) > 1e-3:
                lmid = 0.5 * (l2 + l1)
                # OC formula: x_new = x * sqrt(-dc / lmid)
                B = -dc / lmid
                B = np.maximum(B, 1e-12)
                x_candidate = x * np.sqrt(B)

                # Apply move limits
                x_new = np.maximum(x_min, np.maximum(x - move,
                         np.minimum(x_max, np.minimum(x + move, x_candidate))))

                # Check volume constraint
                vol = np.sum(x_new * elem_volumes) / np.sum(elem_volumes)
                if vol > vol_target:
                    l1 = lmid
                else:
                    l2 = lmid

            return x_new

        # ====== Main optimization loop ======
        x = np.full(n_elems, vf)  # uniform initial density
        self.compliance_history = []

        E_diff = E_fiber - E_base

        print(f"[SIMP] Starting optimization: vf={vf}, p={p}, "
              f"E_fiber={E_fiber:.0f}, E_base={E_base:.0f}")
        print(f"[SIMP] filter_radius={filter_radius}, max_iter={max_iter}, tol={tol}")

        prev_compliance = np.inf

        for it in range(max_iter):
            # 1. Effective Young's modulus per element
            E_eff = E_base + x ** p * E_diff  # (n_elems,)

            # 2. Assemble and solve
            K = _assemble_scaled_K(E_eff)
            u = solve(*condense(K, f_vec, D=fixed_dofs))

            # 3. Element strain energies
            ce = _element_strain_energy(u, E_eff)

            # Total compliance
            compliance = f_vec.dot(u)
            self.compliance_history.append(compliance)

            # 4. Sensitivity: dc/dx_e = -p * x^(p-1) * (E_fiber - E_base) * ce_e / E_eff_e
            # Since ce_e is computed with E_eff, the actual sensitivity for unit E is:
            # dc/dx_e = -p * x^(p-1) * E_diff * (ce_e / E_eff_e)
            # Simplified: using the standard SIMP sensitivity formula
            dc = -p * x ** (p - 1) * E_diff * ce / np.maximum(E_eff, 1e-6)

            # 5. Filter sensitivities
            dc = _density_filter(dc, x)

            # 6. OC update
            x_new = _oc_update(x, dc, vf)

            # 7. Convergence check
            change = np.max(np.abs(x_new - x))
            rel_change = abs(compliance - prev_compliance) / max(abs(prev_compliance), 1e-12)

            vol_frac = np.sum(x_new * elem_volumes) / np.sum(elem_volumes)

            print(f"  iter {it+1:3d}: C={compliance:.4f}, "
                  f"vol={vol_frac:.4f}, change={change:.4f}")

            x = x_new
            prev_compliance = compliance

            if change < tol and it > 5:
                print(f"[SIMP] Converged at iteration {it+1} (change={change:.6f} < {tol})")
                break

        # Store results
        self.densities = x
        fiber_region = x > 0.5

        n_fiber = np.sum(fiber_region)
        print(f"[SIMP] Done: {n_fiber}/{n_elems} elements in fiber region "
              f"({n_fiber/n_elems:.1%})")

        return {
            "densities": x,
            "compliance_history": self.compliance_history,
            "centroids": self.centroids,
            "fiber_region": fiber_region,
            "n_iterations": min(it + 1, max_iter),
        }

    def get_fiber_region(self, threshold: float = 0.5) -> NDArray:
        """获取纤维放置区域 (density > threshold 的单元)

        Args:
            threshold: 密度阈值, 默认 0.5

        Returns:
            (N,) bool mask, True = 放纤维
        """
        if self.densities is None:
            raise RuntimeError("先调用 optimize()")
        return self.densities > threshold

    def export_density_field(self, output_path: str) -> str:
        """导出密度场为 NPZ 文件

        Args:
            output_path: 输出 .npz 路径

        Returns:
            输出路径
        """
        if self.densities is None:
            raise RuntimeError("先调用 optimize()")

        np.savez_compressed(
            output_path,
            densities=self.densities,
            centroids=self.centroids,
            fiber_region=self.get_fiber_region(),
            compliance_history=np.array(self.compliance_history),
            volume_fraction=self.volume_fraction,
            penalty=self.penalty,
        )
        print(f"[SIMP] Density field exported to {output_path}")
        return output_path

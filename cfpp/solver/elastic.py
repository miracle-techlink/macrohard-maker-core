"""线弹性FEA求解器 (scikit-fem) — 向量化版

支持:
  - 各向同性材料 (isotropic)
  - 正交各向异性 / 横观各向同性材料 (transversely isotropic)
    适用于连续碳纤维增强复合材料, 纤维取向逐单元变化
"""

import numpy as np
from numpy.typing import NDArray
from pathlib import Path

from skfem import *
from skfem.models.elasticity import linear_elasticity, lame_parameters
from skfem.io.meshio import from_meshio
import meshio

from cfpp.solver.orthotropic import (
    stiffness_matrix,
    rotate_stiffness_batch,
    direction_to_angle,
    voigt_strain_3d,
    voigt_stress_from_strain,
    CF_PA6_DEFAULTS,
)


class ElasticSolver:
    """线弹性求解器，基于 scikit-fem"""

    def __init__(self, mesh_path: str):
        self.mesh_path = Path(mesh_path)
        self._load_mesh()
        self.E = 210e3
        self.nu = 0.3
        self.displacement = None
        self.stress_tensor = None
        self._basis = None
        # 正交各向异性材料参数
        self._is_orthotropic = False
        self.E1 = None
        self.E2 = None
        self.G12 = None
        self.nu12 = None
        self.nu23 = None
        self._C_material = None  # 材料坐标系下的 6x6 刚度矩阵

    def _load_mesh(self):
        mio = meshio.read(str(self.mesh_path))
        self.mesh_sk = from_meshio(mio)
        n = self.mesh_sk.p.shape[1]
        t = self.mesh_sk.t.shape[1]
        print(f"Mesh: {n} nodes, {t} elements")

    def set_isotropic_material(self, E: float, nu: float):
        self.E = E
        self.nu = nu
        self._is_orthotropic = False

    def set_orthotropic_material(
        self,
        E1: float = CF_PA6_DEFAULTS["E1"],
        E2: float = CF_PA6_DEFAULTS["E2"],
        G12: float = CF_PA6_DEFAULTS["G12"],
        nu12: float = CF_PA6_DEFAULTS["nu12"],
        nu23: float = CF_PA6_DEFAULTS["nu23"],
    ):
        """设置正交各向异性材料参数 (横观各向同性).

        默认值为典型 CF/PA6 连续碳纤维复合材料参数.

        Args:
            E1: 纤维方向弹性模量 (MPa)
            E2: 垂直纤维方向弹性模量 (MPa)
            G12: 面内剪切模量 (MPa)
            nu12: 主泊松比 (fiber-transverse)
            nu23: 横向泊松比 (transverse-transverse)
        """
        self.E1 = E1
        self.E2 = E2
        self.G12 = G12
        self.nu12 = nu12
        self.nu23 = nu23
        self._is_orthotropic = True
        # 预计算材料坐标系下的刚度矩阵
        self._C_material = stiffness_matrix(E1, E2, G12, nu12, nu23)
        print(f"Orthotropic material set: E1={E1}, E2={E2}, G12={G12}, "
              f"nu12={nu12}, nu23={nu23}")

    def solve_with_fiber_orientation(
        self,
        fiber_dirs: NDArray,
        fixed_boundary: str = "fixed",
        load_boundary: str = "load",
        traction: tuple[float, float, float] = (0, 0, -10.0),
    ) -> None:
        """使用逐单元纤维取向求解正交各向异性弹性问题.

        通过自定义双线性形式组装刚度矩阵, 每个单元根据局部纤维方向
        旋转刚度张量.

        Args:
            fiber_dirs: 纤维方向, 可以是:
                - (n_elements,) 角度数组 (弧度), 绕 z 轴旋转
                - (n_elements, 2) 或 (n_elements, 3) 方向向量
            fixed_boundary: 固定边界名称
            load_boundary: 载荷边界名称
            traction: 面力向量 (fx, fy, fz) MPa
        """
        if not self._is_orthotropic:
            raise RuntimeError("先调用 set_orthotropic_material()")

        m = self.mesh_sk
        e = ElementVector(ElementTetP1(), 3)
        ib = Basis(m, e)
        self._basis = ib
        n_elements = m.t.shape[1]

        # ---------- 处理纤维方向输入 ----------
        fiber_dirs = np.asarray(fiber_dirs, dtype=float)
        if fiber_dirs.ndim == 1:
            thetas = fiber_dirs  # 已经是角度
        elif fiber_dirs.ndim == 2:
            thetas = direction_to_angle(fiber_dirs)
        else:
            raise ValueError(f"fiber_dirs shape {fiber_dirs.shape} not supported. "
                             "Expected (N,) angles or (N, 2/3) direction vectors.")

        if len(thetas) != n_elements:
            raise ValueError(f"fiber_dirs length {len(thetas)} != n_elements {n_elements}")

        # ---------- 批量旋转刚度矩阵 ----------
        C_all = rotate_stiffness_batch(self._C_material, thetas, axis="z")
        # C_all: (n_elements, 6, 6)

        print(f"Assembling orthotropic stiffness ({n_elements} elements)...")

        # ---------- 自定义双线性形式 ----------
        # 将 C_all 映射到积分点: 每个四面体单元的每个积分点共享该单元的 C
        # 在 TetP1 基上, 积分点数取决于 quadrature; 通常 1 点高斯积分
        # 我们将 C 存入 basis 的额外数据

        # 获取每个积分点对应的单元索引, 并构造 C 在积分点上的值
        # ib.dx shape: (n_integration_pts_per_elem, n_elements)
        n_qp = ib.X.shape[-2] if ib.X.ndim > 2 else 1

        # C at quadrature points: (6, 6, n_qp, n_elements) — 广播单元值到所有积分点
        C_qp = np.broadcast_to(
            C_all.transpose(1, 2, 0)[..., np.newaxis, :],   # (6,6,1,n_elem)
            (6, 6, n_qp, n_elements) if n_qp > 1 else (6, 6, n_elements)
        )
        if C_qp.ndim == 3:
            # 只有1个积分点: shape (6, 6, n_elements)
            C_qp = C_qp[:, :, np.newaxis, :]  # -> (6, 6, 1, n_elements)

        @BilinearForm
        def orthotropic_stiffness(u, v, w):
            """正交各向异性弹性双线性形式.

            在 Voigt 记号下: a(u,v) = integral eps(v)^T C eps(u) dV
            """
            # u.grad, v.grad: (3, 3, *) 其中 * = 积分点形状
            # 构造 Voigt 应变
            def sym_grad_voigt(field):
                """field.grad -> 6-component Voigt strain."""
                g = field.grad
                return np.array([
                    g[0, 0],                        # e11
                    g[1, 1],                        # e22
                    g[2, 2],                        # e33
                    g[1, 2] + g[2, 1],              # 2*e23
                    g[0, 2] + g[2, 0],              # 2*e13
                    g[0, 1] + g[1, 0],              # 2*e12
                ])  # shape: (6, *integration_shape)

            eps_u = sym_grad_voigt(u)  # (6, n_qp, n_elem) or (6, n_elem)
            eps_v = sym_grad_voigt(v)

            # 确保形状兼容
            if eps_u.ndim == 2:
                eps_u = eps_u[:, np.newaxis, :]
                eps_v = eps_v[:, np.newaxis, :]

            # sigma = C @ eps_u:  C is (6, 6, n_qp, n_elem), eps_u is (6, n_qp, n_elem)
            sigma = np.einsum('ijqe,jqe->iqe', w.C, eps_u)

            # eps_v^T @ sigma = sum over i
            return np.einsum('iqe,iqe->qe', eps_v, sigma)

        K = orthotropic_stiffness.assemble(ib, C=C_qp)

        # ---------- 载荷 ----------
        traction = np.array(traction, dtype=float)

        if load_boundary in m.boundaries:
            load_facets = m.boundaries[load_boundary]
        else:
            load_facets = m.facets_satisfying(
                lambda x: np.abs(x[0] - np.max(m.p[0])) < 1e-6
            )

        fb = FacetBasis(m, e, facets=load_facets)

        @LinearForm
        def surface_load(v, w):
            return (traction[0] * v.value[0]
                    + traction[1] * v.value[1]
                    + traction[2] * v.value[2])

        f = surface_load.assemble(fb)

        # ---------- 边界条件 ----------
        if fixed_boundary in m.boundaries:
            fixed_facets = m.boundaries[fixed_boundary]
        else:
            fixed_facets = m.facets_satisfying(
                lambda x: np.abs(x[0]) < 1e-6
            )

        fixed_dofs = ib.get_dofs(fixed_facets)
        all_fixed = fixed_dofs.all()

        # ---------- 求解 ----------
        self.displacement = solve(*condense(K, f, D=all_fixed))
        max_disp = np.max(np.abs(self.displacement))
        print(f"Orthotropic solve done. max|u| = {max_disp:.6f} mm")

        # 保存 C_all 供后续应力计算
        self._C_elements = C_all

    def compute_stress_orthotropic(self) -> None:
        """计算正交各向异性材料的应力张量 (逐单元变刚度)."""
        if self.displacement is None:
            raise RuntimeError("先调用 solve_with_fiber_orientation()")
        if not hasattr(self, '_C_elements') or self._C_elements is None:
            raise RuntimeError("无正交各向异性刚度数据, 请先调用 solve_with_fiber_orientation()")

        ib = self._basis
        du = ib.interpolate(self.displacement)
        # grad_u: (3, 3, n_qp, n_elem) 或 (3, 3, n_elem)
        grad_u_raw = du.grad
        if grad_u_raw.ndim == 4:
            grad_u = grad_u_raw.mean(axis=2)  # 对积分点取平均 -> (3, 3, n_elem)
        else:
            grad_u = grad_u_raw  # (3, 3, n_elem)

        n_elem = grad_u.shape[2]

        # Voigt 应变: (n_elem, 6)
        eps = np.zeros((n_elem, 6))
        eps[:, 0] = grad_u[0, 0]
        eps[:, 1] = grad_u[1, 1]
        eps[:, 2] = grad_u[2, 2]
        eps[:, 3] = grad_u[1, 2] + grad_u[2, 1]
        eps[:, 4] = grad_u[0, 2] + grad_u[2, 0]
        eps[:, 5] = grad_u[0, 1] + grad_u[1, 0]

        # sigma = C @ eps, element-wise
        sigma_voigt = voigt_stress_from_strain(self._C_elements, eps)  # (n_elem, 6)

        # 转换回 3x3 张量
        stress = np.zeros((n_elem, 3, 3))
        stress[:, 0, 0] = sigma_voigt[:, 0]  # s11
        stress[:, 1, 1] = sigma_voigt[:, 1]  # s22
        stress[:, 2, 2] = sigma_voigt[:, 2]  # s33
        stress[:, 1, 2] = sigma_voigt[:, 3]  # s23
        stress[:, 2, 1] = sigma_voigt[:, 3]
        stress[:, 0, 2] = sigma_voigt[:, 4]  # s13
        stress[:, 2, 0] = sigma_voigt[:, 4]
        stress[:, 0, 1] = sigma_voigt[:, 5]  # s12
        stress[:, 1, 0] = sigma_voigt[:, 5]

        self.stress_tensor = stress
        print("Orthotropic stress tensor computed.")

    def solve(
        self,
        fixed_boundary: str = "fixed",
        load_boundary: str = "load",
        traction: tuple[float, float, float] = (0, 0, -10.0),
    ) -> None:
        m = self.mesh_sk
        e = ElementVector(ElementTetP1(), 3)
        ib = Basis(m, e)
        self._basis = ib

        lam, mu = lame_parameters(self.E, self.nu)
        K = linear_elasticity(lam, mu).assemble(ib)

        traction = np.array(traction, dtype=float)

        if load_boundary in m.boundaries:
            load_facets = m.boundaries[load_boundary]
        else:
            load_facets = m.facets_satisfying(
                lambda x: np.abs(x[0] - np.max(m.p[0])) < 1e-6
            )

        fb = FacetBasis(m, e, facets=load_facets)

        @LinearForm
        def surface_load(v, w):
            return traction[0] * v.value[0] + traction[1] * v.value[1] + traction[2] * v.value[2]

        f = surface_load.assemble(fb)

        if fixed_boundary in m.boundaries:
            fixed_facets = m.boundaries[fixed_boundary]
        else:
            fixed_facets = m.facets_satisfying(
                lambda x: np.abs(x[0]) < 1e-6
            )

        fixed_dofs = ib.get_dofs(fixed_facets)
        all_fixed = fixed_dofs.all()

        self.displacement = solve(*condense(K, f, D=all_fixed))
        max_disp = np.max(np.abs(self.displacement))
        print(f"Solved. max|u| = {max_disp:.6f} mm")

    def compute_stress(self) -> None:
        """向量化应力张量计算"""
        if self.displacement is None:
            raise RuntimeError("先调用 solve()")

        ib = self._basis
        lam, mu = lame_parameters(self.E, self.nu)

        du = ib.interpolate(self.displacement)
        grad_u = du.grad.mean(axis=-1)  # (3, 3, n_cells)

        eps = 0.5 * (grad_u + np.swapaxes(grad_u, 0, 1))
        tr_eps = eps[0, 0] + eps[1, 1] + eps[2, 2]

        sigma = 2 * mu * eps
        for i in range(3):
            sigma[i, i] += lam * tr_eps

        self.stress_tensor = np.transpose(sigma, (2, 0, 1))  # (n_cells, 3, 3)
        print("Stress tensor computed.")

    def extract_principal_stresses(self) -> dict:
        """向量化主应力提取 (批量 eigh)"""
        if self.stress_tensor is None:
            self.compute_stress()

        m = self.mesh_sk
        n_cells = m.t.shape[1]

        # 向量化 centroids
        centroids = m.p[:, m.t].mean(axis=1).T  # (n_cells, 3)

        # 对称化
        S = 0.5 * (self.stress_tensor + np.swapaxes(self.stress_tensor, 1, 2))

        # 批量特征分解: np.linalg.eigh 支持 (..., 3, 3) 输入
        eigenvalues, eigenvectors = np.linalg.eigh(S)  # vals: (N,3), vecs: (N,3,3)

        # eigh 返回升序, 翻转为降序
        eigenvalues = eigenvalues[:, ::-1]
        eigenvectors = eigenvectors[:, :, ::-1]

        s1, s2, s3 = eigenvalues[:, 0], eigenvalues[:, 1], eigenvalues[:, 2]
        von_mises = np.sqrt(0.5 * ((s1 - s2)**2 + (s2 - s3)**2 + (s3 - s1)**2))

        result = {
            "centroids": centroids,
            "sigma_1_val": s1,
            "sigma_2_val": s2,
            "sigma_3_val": s3,
            "sigma_1_dir": eigenvectors[:, :, 0],  # (N, 3)
            "sigma_2_dir": eigenvectors[:, :, 1],
            "sigma_3_dir": eigenvectors[:, :, 2],
            "von_mises": von_mises,
        }

        print(f"Principal stresses: {n_cells} elements")
        print(f"  σ1: [{s1.min():.2f}, {s1.max():.2f}] MPa")
        print(f"  von Mises max: {von_mises.max():.2f} MPa")

        return result

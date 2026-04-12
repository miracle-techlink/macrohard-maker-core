"""正交各向异性 (横观各向同性) 材料模型

提供刚度张量计算、Bond 矩阵旋转等工具函数，
用于连续碳纤维增强复合材料的有限元分析。

坐标约定:
  1-方向 = 纤维方向 (高模量)
  2,3-方向 = 横向 (各向同性平面)

参考: Jones, "Mechanics of Composite Materials", 2nd ed.
"""

import numpy as np
from numpy.typing import NDArray


# ---------- 典型 CF/PA6 材料参数 ----------
CF_PA6_DEFAULTS = dict(
    E1=60000.0,    # MPa, 纤维方向
    E2=3500.0,     # MPa, 横向
    G12=2000.0,    # MPa, 面内剪切
    nu12=0.3,      # 主泊松比
    nu23=0.45,     # 横向泊松比
)


def compliance_matrix(E1: float, E2: float, G12: float,
                      nu12: float, nu23: float) -> NDArray:
    """构建 6x6 柔度矩阵 S (Voigt 记号).

    横观各向同性: 2-3 平面各向同性, 1 为纤维方向.

    Voigt 顺序: [11, 22, 33, 23, 13, 12]

    Returns:
        S: (6, 6) compliance matrix, units 1/MPa
    """
    nu21 = nu12 * E2 / E1  # 对称性: nu21/E2 = nu12/E1
    G23 = E2 / (2.0 * (1.0 + nu23))

    S = np.zeros((6, 6))
    # 正应力 - 正应变
    S[0, 0] = 1.0 / E1
    S[1, 1] = 1.0 / E2
    S[2, 2] = 1.0 / E2

    # 耦合项
    S[0, 1] = -nu12 / E1
    S[1, 0] = -nu12 / E1   # = -nu21/E2
    S[0, 2] = -nu12 / E1
    S[2, 0] = -nu12 / E1
    S[1, 2] = -nu23 / E2
    S[2, 1] = -nu23 / E2

    # 剪切
    S[3, 3] = 1.0 / G23
    S[4, 4] = 1.0 / G12
    S[5, 5] = 1.0 / G12

    return S


def stiffness_matrix(E1: float, E2: float, G12: float,
                     nu12: float, nu23: float) -> NDArray:
    """构建 6x6 刚度矩阵 C = inv(S).

    Returns:
        C: (6, 6) stiffness matrix, units MPa
    """
    S = compliance_matrix(E1, E2, G12, nu12, nu23)
    C = np.linalg.inv(S)
    return C


def bond_rotation_matrix(theta: float, axis: str = "z") -> NDArray:
    """构建 Bond 变换矩阵 T (6x6), 用于将 Voigt 刚度张量从材料坐标系旋转到全局坐标系.

    绕指定轴旋转角度 theta (弧度), 使材料 1-方向 (纤维) 从全局 x 轴
    旋转到指定方向.

    对于 2D 打印平面内的纤维取向, 通常绕 z 轴旋转.

    Voigt 顺序: [11, 22, 33, 23, 13, 12]

    Args:
        theta: 旋转角 (弧度), 纤维方向与全局 x 轴的夹角
        axis: 旋转轴, "z" (默认, 平面内旋转) 或 "y"

    Returns:
        T: (6, 6) Bond transformation matrix
    """
    c = np.cos(theta)
    s = np.sin(theta)

    if axis == "z":
        # 绕 z 轴: 1-2 平面内旋转, 3-方向不变
        T = np.array([
            [c*c,    s*s,    0,  0,      0,      2*c*s  ],
            [s*s,    c*c,    0,  0,      0,     -2*c*s  ],
            [0,      0,      1,  0,      0,      0      ],
            [0,      0,      0,  c,     -s,      0      ],
            [0,      0,      0,  s,      c,      0      ],
            [-c*s,   c*s,    0,  0,      0,      c*c-s*s],
        ])
    elif axis == "y":
        # 绕 y 轴: 1-3 平面内旋转, 2-方向不变
        T = np.array([
            [c*c,    0,  s*s,    0,      2*c*s,  0],
            [0,      1,  0,      0,      0,      0],
            [s*s,    0,  c*c,    0,     -2*c*s,  0],
            [0,      0,  0,      c,      0,      s],
            [-c*s,   0,  c*s,    0,      c*c-s*s,0],
            [0,      0,  0,     -s,      0,      c],
        ])
    else:
        raise ValueError(f"Unsupported axis: {axis}. Use 'z' or 'y'.")

    return T


def rotate_stiffness(C_mat: NDArray, theta: float, axis: str = "z") -> NDArray:
    """将材料坐标系下的刚度矩阵 C 旋转到全局坐标系.

    C_global = T^T @ C_material @ T

    Args:
        C_mat: (6, 6) 材料坐标系刚度矩阵
        theta: 旋转角 (弧度)
        axis: 旋转轴

    Returns:
        C_global: (6, 6) 全局坐标系刚度矩阵
    """
    T = bond_rotation_matrix(theta, axis)
    return T.T @ C_mat @ T


def rotate_stiffness_batch(C_mat: NDArray, thetas: NDArray,
                           axis: str = "z") -> NDArray:
    """批量旋转刚度矩阵 (向量化).

    Args:
        C_mat: (6, 6) 材料坐标系刚度矩阵 (所有单元共享)
        thetas: (N,) 每个单元的纤维角度 (弧度)
        axis: 旋转轴

    Returns:
        C_rotated: (N, 6, 6) 每个单元旋转后的刚度矩阵
    """
    N = len(thetas)
    c = np.cos(thetas)
    s = np.sin(thetas)

    C_rotated = np.zeros((N, 6, 6))

    if axis == "z":
        # 向量化构建 N 个 Bond 矩阵
        T = np.zeros((N, 6, 6))
        c2 = c * c
        s2 = s * s
        cs = c * s
        c2_s2 = c2 - s2

        T[:, 0, 0] = c2;    T[:, 0, 1] = s2;    T[:, 0, 5] = 2*cs
        T[:, 1, 0] = s2;    T[:, 1, 1] = c2;    T[:, 1, 5] = -2*cs
        T[:, 2, 2] = 1.0
        T[:, 3, 3] = c;     T[:, 3, 4] = -s
        T[:, 4, 3] = s;     T[:, 4, 4] = c
        T[:, 5, 0] = -cs;   T[:, 5, 1] = cs;    T[:, 5, 5] = c2_s2

        # C_global = T^T @ C_mat @ T,  批量矩阵乘法
        # T: (N,6,6), C_mat: (6,6)
        TC = np.einsum('nij,jk->nik', T, C_mat)       # (N,6,6)
        C_rotated = np.einsum('nji,njk->nik', T, TC)   # T^T @ (T @ C) 错, 需要 T^T @ C @ T
        # 修正: T^T @ C_mat @ T
        TtC = np.einsum('nji,jk->nik', T, C_mat)       # T^T @ C_mat: (N,6,6)
        C_rotated = np.einsum('nij,njk->nik', TtC, T)   # (T^T @ C_mat) @ T: (N,6,6)
    else:
        # 非 z 轴: 逐个旋转 (通常用不到批量)
        for i in range(N):
            C_rotated[i] = rotate_stiffness(C_mat, thetas[i], axis)

    return C_rotated


def direction_to_angle(fiber_dirs: NDArray) -> NDArray:
    """将纤维方向向量转换为绕 z 轴的角度.

    Args:
        fiber_dirs: (N, 2) 或 (N, 3) 纤维方向向量 (只用 x,y 分量)

    Returns:
        thetas: (N,) 角度 (弧度), 范围 [-pi, pi]
    """
    fiber_dirs = np.asarray(fiber_dirs)
    return np.arctan2(fiber_dirs[:, 1], fiber_dirs[:, 0])


def voigt_strain_3d(grad_u: NDArray) -> NDArray:
    """从位移梯度计算 Voigt 应变向量.

    Args:
        grad_u: (..., 3, 3) 位移梯度 du_i/dx_j

    Returns:
        strain: (..., 6) Voigt 应变 [e11, e22, e33, 2*e23, 2*e13, 2*e12]
    """
    e = np.zeros(grad_u.shape[:-2] + (6,))
    e[..., 0] = grad_u[..., 0, 0]                          # e11
    e[..., 1] = grad_u[..., 1, 1]                          # e22
    e[..., 2] = grad_u[..., 2, 2]                          # e33
    e[..., 3] = grad_u[..., 1, 2] + grad_u[..., 2, 1]     # 2*e23
    e[..., 4] = grad_u[..., 0, 2] + grad_u[..., 2, 0]     # 2*e13
    e[..., 5] = grad_u[..., 0, 1] + grad_u[..., 1, 0]     # 2*e12
    return e


def voigt_stress_from_strain(C: NDArray, strain: NDArray) -> NDArray:
    """sigma = C @ epsilon (Voigt).

    Args:
        C: (6, 6) 或 (N, 6, 6) 刚度矩阵
        strain: (N, 6) Voigt 应变

    Returns:
        stress: (N, 6) Voigt 应力
    """
    if C.ndim == 2:
        return strain @ C.T   # (N,6) @ (6,6)^T = (N,6)
    else:
        # element-wise: C is (N,6,6), strain is (N,6)
        return np.einsum('nij,nj->ni', C, strain)

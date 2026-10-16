"""Transparent reference discretizations; these are classical controls, not inventions."""

from dataclasses import dataclass
import numpy as np
from scipy.linalg import solve


@dataclass
class Heat1D:
    x: np.ndarray
    edges_k: np.ndarray
    source: np.ndarray
    matrix: np.ndarray
    rhs: np.ndarray
    boundary: tuple

    @property
    def h(self):
        return 1 / (len(self.x) + 1)

    def reference(self):
        return solve(self.matrix, self.rhs, assume_a="pos")

    def flux(self, temperature):
        full = np.r_[self.boundary[0], temperature, self.boundary[1]]
        return -self.edges_k * np.diff(full) / self.h

    def diagnostics(self, temperature):
        flux = self.flux(temperature)
        residual = self.matrix @ temperature - self.rhs
        return {
            "pde_residual_inf": float(np.max(np.abs(residual))),
            "relative_residual": float(np.linalg.norm(residual) / max(np.linalg.norm(self.rhs), 1e-14)),
            "net_energy_balance_abs": float(abs(flux[-1] - flux[0] - self.h * self.source.sum())),
            "boundary_error_abs": 0.0,  # boundary values are prescribed, not predicted
        }


def heat_1d(n=15, conductivity=1.0, source=None, boundary=(0.0, 0.0)):
    h = 1 / (n + 1)
    x = np.arange(1, n + 1) * h
    edge_x = (np.arange(n + 1) + 0.5) * h
    k = np.asarray(conductivity(edge_x) if callable(conductivity) else conductivity)
    k = np.broadcast_to(k, (n + 1,)).copy()
    if not np.all(k > 0):
        raise ValueError("conductivity must be positive")
    q = np.asarray(source(x) if callable(source) else (np.ones(n) if source is None else source), dtype=float)
    q = np.broadcast_to(q, (n,)).copy()
    matrix = np.diag((k[:-1] + k[1:]) / h**2)
    matrix += np.diag(-k[1:-1] / h**2, 1) + np.diag(-k[1:-1] / h**2, -1)
    rhs = q.copy()
    rhs[0] += k[0] / h**2 * boundary[0]
    rhs[-1] += k[-1] / h**2 * boundary[1]
    return Heat1D(x, k, q, matrix, rhs, boundary)


def neumann_1d(n=16, source=None, left_flux=0.0, right_flux=0.0):
    """Cell-centered unit-conductivity problem, fixed mean zero and compatible flux."""
    h = 1/n
    x = (np.arange(n) + 0.5)*h
    q = np.cos(2*np.pi*x)*(2*np.pi)**2 if source is None else np.broadcast_to(source, (n,)).copy()
    if not np.isclose(q.sum()*h, right_flux-left_flux, atol=1e-10):
        raise ValueError("incompatible pure Neumann source and outward-directed x fluxes")
    d = np.diag(np.ones(n-1), 1) + np.diag(np.ones(n-1), -1)
    matrix = -d + np.diag(np.r_[1., np.full(n-2, 2.), 1.])
    matrix /= h**2
    rhs = q.copy()
    rhs[0] += left_flux/h
    rhs[-1] -= right_flux/h
    augmented = np.block([[matrix, np.ones((n,1))], [np.ones((1,n)), np.zeros((1,1))]])
    solution = solve(augmented, np.r_[rhs, 0.], assume_a="sym")[:n]
    return x, matrix, rhs, solution


@dataclass
class Heat2D:
    points: np.ndarray
    triangles: np.ndarray
    tensors: np.ndarray
    stiffness: np.ndarray
    load: np.ndarray
    mass: np.ndarray
    free: np.ndarray
    boundary_values: np.ndarray
    matrix: np.ndarray
    rhs: np.ndarray
    lift: np.ndarray

    def decode(self, z):
        full = self.lift.copy()
        full[self.free] = z / np.sqrt(self.mass[self.free])
        return full

    def reference(self):
        return self.decode(solve(self.matrix, self.rhs, assume_a="pos"))

    def flux(self, full):
        fluxes = []
        for tri, tensor in zip(self.triangles, self.tensors):
            coefficients = np.linalg.solve(np.c_[np.ones(3), self.points[tri]], full[tri])
            fluxes.append(-tensor @ coefficients[1:])
        return np.array(fluxes)

    def diagnostics(self, full):
        residual = self.stiffness @ full - self.load
        boundary = np.setdiff1d(np.arange(len(full)), self.free)
        return {
            "pde_residual_inf": float(np.max(np.abs(residual[self.free]))),
            "relative_residual": float(np.linalg.norm(residual[self.free]) / max(np.linalg.norm(self.load[self.free]), 1e-14)),
            "net_energy_balance_abs": float(abs(-residual[boundary].sum() - self.load.sum())),
            "boundary_error_abs": float(np.max(np.abs(full[boundary]-self.boundary_values[boundary]))),
            "source_integral": float(self.load.sum()),
            "outward_flux_from_reactions": float(-residual[boundary].sum()),
        }


def heat_2d(intervals=16, tensor=None, source=None, boundary=None, hole=False, aspect=1.0):
    """Linear triangular FEM, consistent with -div(K grad T)=Q.

    Mass scaling gives a symmetric L2-coordinate operator for spectral experiments.
    Holes are grid-aligned squares, not arbitrary CAD geometry.
    """
    m = intervals
    points = np.array([(aspect*i/m, j/m) for j in range(m+1) for i in range(m+1)])
    triangles = []
    for j in range(m):
        for i in range(m):
            ids = [j*(m+1)+i, j*(m+1)+i+1, (j+1)*(m+1)+i, (j+1)*(m+1)+i+1]
            for tri in [(ids[0],ids[1],ids[3]), (ids[0],ids[3],ids[2])]:
                center = points[list(tri)].mean(axis=0)
                if hole and 0.375*aspect < center[0] < 0.625*aspect and 0.375 < center[1] < 0.625:
                    continue
                triangles.append(tri)
    used = np.unique(triangles)
    remap = np.full(len(points), -1, dtype=int)
    remap[used] = np.arange(len(used))
    points = points[used]
    triangles = remap[np.array(triangles)]
    n = len(points)
    stiffness, load, mass = np.zeros((n,n)), np.zeros(n), np.zeros(n)
    edge_counts = {}
    tensors = []
    for tri in triangles:
        vertices = points[tri]
        center = vertices.mean(axis=0)
        shape = np.c_[np.ones(3), vertices]
        area = abs(np.linalg.det(shape))/2
        gradients = np.linalg.inv(shape)[1:, :]
        k = np.eye(2) if tensor is None else np.asarray(tensor(*center) if callable(tensor) else tensor)
        if not np.allclose(k, k.T) or np.linalg.eigvalsh(k).min() <= 0:
            raise ValueError("conductivity tensor must be symmetric positive definite")
        tensors.append(k)
        stiffness[np.ix_(tri,tri)] += area * gradients.T @ k @ gradients
        q = 1.0 if source is None else source(*center)
        load[tri] += area*q/3
        mass[tri] += area/3
        for a,b in [(tri[0],tri[1]), (tri[1],tri[2]), (tri[2],tri[0])]:
            edge=tuple(sorted((a,b)))
            edge_counts[edge]=edge_counts.get(edge,0)+1
    boundary_ids = np.unique([edge for edge,count in edge_counts.items() if count == 1])
    free = np.setdiff1d(np.arange(n), boundary_ids)
    values = np.zeros(n) if boundary is None else np.array([boundary(x,y) for x,y in points])
    lift = np.zeros(n)
    lift[boundary_ids] = values[boundary_ids]
    scale = np.sqrt(mass[free])
    matrix = stiffness[np.ix_(free,free)] / scale[:,None] / scale[None,:]
    rhs = (load-stiffness @ lift)[free]/scale
    return Heat2D(points, triangles, np.array(tensors), stiffness, load, mass, free, values, matrix, rhs, lift)

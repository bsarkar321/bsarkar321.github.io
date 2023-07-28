import numba
from numba import jit
from numba.experimental import jitclass
import numpy as np
import matplotlib.pyplot as plt

d = 2

n = 80

dt = 1e-4
frame_dt = 1e-2
dx = 1/n
inv_dx = 1/dx

gravity = np.array([0, -200] + [0] * (d-1))
boundary = 0.05

mass = 1
vol = 1
hardening = 10
E = 1e4
nu = 0.2
plastic = True

mu_0 = E / (2 * (1 + nu))
lam_0 = E * nu / ((1 + nu) * (1 - 2 * nu))


@jit(nopython=True)
def polar(m):
    x = m[0, 0] + m[1, 1]
    y = m[1, 0] - m[0, 1]
    scale = 1 / np.sqrt(x * x + y * y)
    c = x * scale
    s = y * scale
    R = np.array([[c, -s], [s, c]])
    S = R.T * m
    return R, S



@jitclass([
    ('x', numba.float64[:]),
    ('v', numba.float64[:]),
    ('F', numba.float64[:, :]),
    ('C', numba.float64[:, :]),
    ('Jp', numba.float64),
    ('c', numba.types.string)
])
class Particle:

    def __init__(self, x, c, v=None, F=None, C=None, Jp=None):
        if v is None:
            v = np.zeros((d,))
        if F is None:
            F = np.eye(d)
        if C is None:
            C = np.zeros((d, d))
        if Jp is None:
            Jp = 1
        self.x = x
        self.v = v
        self.F = F
        self.C = C
        self.Jp = Jp
        self.c = c


@jit(nopython=True)
def advance(dt, particles, grid):
    # Reset Grid
    grid.fill(0)
    # P2G
    for p in particles:
        base = np.floor(p.x * inv_dx - 0.5)
        fx = p.x * inv_dx - base

        # Quadratic kernel (eqn 12)
        w = [
            0.5 * ((1.5 - fx) ** 2),
            0.75 - (fx - 1) ** 2,
            0.5 * ((fx - 0.5) ** 2)
        ]

        # Lame parameters for snow (86)
        e = np.exp(hardening * (1 - p.Jp))
        mu = mu_0 * e
        lam = lam_0 * e

        # Volume
        J = np.linalg.det(p.F)

        # Polar decomp for fixed corotated model
        r, s = polar(p.F)

        Dinv = 4 * inv_dx * inv_dx
        # (52)
        PF = 2 * mu * (p.F - r) @ p.F.T + lam * (J-1) * J * np.eye(d)

        # Cauchy stress
        stress = - dt * vol * Dinv * PF

        # Fused APIC momentum + MLS-MPM stress contribution
        affine = stress + mass * p.C

        # P2G Transfer
        for posx in range(3):
            for posy in range(3):
                pos = (posx, posy)
                dpos = (np.array(pos) - fx) * dx

                # Translational momentum
                mass_x_velocity = np.append(p.v * mass, mass)
                weight = 1
                for i in range(d):
                    weight *= w[pos[i]][i]
                gridloc = (base + np.array(pos)).astype(np.int32)
                grid[gridloc[0], gridloc[1]] += (
                    weight * (mass_x_velocity + np.append(affine @ dpos, 0))
                )

    # Grid Node Updates
    for posx in range(n+1):
        for posy in range(n+1):
            pos = (posx, posy)
            g = grid[pos]

            if (g[-1] > 0):
                g /= g[-1]

                g += dt * gravity
                coord = np.array(pos) / n

                if coord[0] < boundary or coord[0] > 1 - boundary or coord[1] > 1 - boundary:
                    g.fill(0)
                for i in coord[2:]:
                    if i < boundary or i > 1 - boundary:
                        g.fill(0)
                        break

                if (coord[1] < boundary):
                    g[1] = max(0, g[1])

    # # G2P
    for p in particles:
        base = np.floor(p.x * inv_dx - 0.5)
        fx = p.x * inv_dx - base

        # Quadratic kernel (eqn 12)
        w = [
            0.5 * ((1.5 - fx) ** 2),
            0.75 - (fx - 1) ** 2,
            0.5 * ((fx - 0.5) ** 2)
        ]

        p.C.fill(0)
        p.v.fill(0)

        for posx in range(3):
            for posy in range(3):
                pos = (posx, posy)
                dpos = (np.array(pos) - fx)
                gridloc = (base + np.array(pos)).astype(np.int32)
                grid_v = grid[gridloc[0], gridloc[1]][:d]
                weight = 1
                for i in range(d):
                    weight *= w[pos[i]][i]
                p.v += weight * grid_v
                p.C += 4 * inv_dx * np.outer(weight * grid_v, dpos)

        # Advection
        p.x += dt * p.v

        # MLS-MPM F-update
        F = (np.eye(d) + dt * p.C) * p.F

        U, sig, V = np.linalg.svd(F)

        # Snow Plasticity
        for i in range(d * plastic):
            if sig[i] < 1.0 - 2.5e-2:
                sig[i] = 1.0 - 2.5e-2
            elif sig[i] > 1.0 + 7.5e-3:
                sig[i] = 1.0 + 7.5e-3

        oldJ = np.linalg.det(F)
        F = (U * sig) @ V

        Jp_new = p.Jp * oldJ / np.linalg.det(F)
        if Jp_new < 0.6:
            Jp_new = 0.6
        elif Jp_new > 20:
            Jp_new = 20.0

        p.Jp = Jp_new
        p.F = F


def add_object(center, c, particles):
    for i in range(1000):
        newpos = (np.random.rand(d) * 2 - 1) * 0.08 + center
        particles.append(Particle(newpos, c))


def plot_points(particles):
    xvals = []
    yvals = []
    colval = []
    for p in particles:
        xvals.append(p.x[0])
        yvals.append(p.x[1])
        colval.append(p.c)
    print(xvals[0])
    print(yvals[0])
    print(len(xvals))
    plt.scatter(xvals, yvals, s=2, c=colval)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.show()
    plt.clf()


def main():
    particles = []
    grid = np.zeros((n + 1, n + 1, d + 1))
    add_object(np.array([0.55, 0.45]), '#ed553b', particles)
    add_object(np.array([0.45, 0.65]), '#f2b134', particles)
    add_object(np.array([0.55, 0.85]), '#068587', particles)

    for step in range(1000):
        advance(dt, particles, grid)

        if step % int(frame_dt / dt) == 0:
            plot_points(particles)


if __name__ == '__main__':
    main()

import glob
import h5py as h5
import json
import matplotlib.pyplot as plt
import numpy as np
import re
import scipy.interpolate
import sys
import synchrad
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm

def get_kpn1():
    try:
        with open('matched/qpinput.json', 'r') as f:
            qpinput_text = f.read()
            qpinput_text = re.sub(r'!.*\n', r'\n', qpinput_text)
            qpinput_text = re.sub(",[ \t\r\n]+}", "}", qpinput_text)
            qpinput_text = re.sub(",[ \t\r\n]+\]", "]", qpinput_text)
            qpinput = json.loads(qpinput_text)
            n0 = qpinput['simulation']['n0']
            kpn1 = 299792458 / np.sqrt(n0 * 100 * 100 * 100 * 1.602176634e-19 * 1.602176634e-19 / (9.109383701528e-31 * 8.854187812813e-12))
    except FileNotFoundError:
        print(f'\033[1;31mError:\033[0m Unable to find qpinput.json, are you in the right directory?')
        sys.exit(1)
    return kpn1
    
def get_trajectories(n_particles='all', seed=0, name='matched'):

    # obtain filenames
    filenames = list(glob.glob(f'{name}/Beam0001/Raw/*'))
    filenames.sort(key=lambda x: int(x[-11:-3]))
    filenames = filenames[:-1]#[:200:20]
    n_files = len(filenames)
    
    # figure out number of particles
    with h5.File(filenames[0], 'r') as file:
        ids = np.array(file['id'])
        if seed is None:
            np.random.default_rng().shuffle(ids)
        else:
            np.random.default_rng(seed).shuffle(ids)
        if n_particles == 'all':
            n_particles = len(ids)
        else:
            assert n_particles <= len(ids)
    assert len(ids) == len(set(ids))
    
    # misc
    kpn1 = get_kpn1()
    z = np.empty((len(filenames)), dtype=np.float64)
    particles = {}
    for i in range(n_particles):
        arr = np.empty((n_files, 6))
        arr[:] = np.nan
        particles[ids[i]] = arr

    # iterate through files
    for i, filename in enumerate(filenames):
        with h5.File(filename, 'r') as file:
            print(f'iterating through file {i+1} of {len(filenames)}')

            # set z
            z[i] = kpn1 * file.attrs['TIME'][0]

            # read particle data
            x = kpn1 * np.array(file['x1'])
            y = kpn1 * np.array(file['x2'])
            zeta = kpn1 * np.array(file['x3'])
            px = np.array(file['p1'])
            py = np.array(file['p2'])
            pz = np.array(file['p3'])
            ids2 = np.array(file['id'])

            # check validity of ids
            #assert len(ids) == len(ids2)

            # append particle data
            for j, id in enumerate(ids2):
                if id in particles:
                    view = particles[id][i,:]
                    view[0] = x[j]
                    view[1] = y[j]
                    view[2] = zeta[j]
                    view[3] = px[j]
                    view[4] = py[j]
                    view[5] = pz[j]

    blacklist = []
    for i, (k, v) in enumerate(particles.items()):
        if np.any(np.isnan(v)):
            blacklist.append(k)

    print(len(particles), 'particles')
    print(len(blacklist), 'blacklisted')

    for k in blacklist:
        del particles[k]

    print(len(particles), 'remaining')
    array = np.empty((len(particles), n_files, 6), dtype=np.float64)
    for i, (k, v) in enumerate(particles.items()):
        array[i, :, :] = v
        
    return z, array


def convert_traj(z, trajectories, points):
    t = z / 299792458
    t2 = np.linspace(t.min(), t.max(), points)
    trajectories2 = np.empty((trajectories.shape[0], points, 9), dtype=np.float64)
    trajectories2[:, :, :3] = scipy.interpolate.make_interp_spline(t, trajectories[:, :, :3], k=3, axis=1)(t2)
    trajectories2[:, :, 2] *= -1
    gamma = np.sqrt(1 + trajectories[:, :, 3] ** 2 + trajectories[:, :, 4] ** 2 + trajectories[:, :, 5] ** 2)
    beta_x = trajectories[:, :, 3] / gamma
    beta_y = trajectories[:, :, 4] / gamma
    beta_x_dot = scipy.interpolate.make_interp_spline(t, beta_x, k=3, axis=1).derivative(nu=1)(t2)
    beta_y_dot = scipy.interpolate.make_interp_spline(t, beta_y, k=3, axis=1).derivative(nu=1)(t2)
    gamma_dot = scipy.interpolate.make_interp_spline(t, gamma, k=3, axis=1).derivative(nu=1)(t2)
    trajectories2[:, :, 3] = scipy.interpolate.make_interp_spline(t, beta_x, k=3, axis=1)(t2)
    trajectories2[:, :, 4] = scipy.interpolate.make_interp_spline(t, beta_y, k=3, axis=1)(t2)
    trajectories2[:, :, 5] = scipy.interpolate.make_interp_spline(t, gamma, k=3, axis=1)(t2)
    trajectories2[:, :, 6] = beta_x_dot
    trajectories2[:, :, 7] = beta_y_dot
    trajectories2[:, :, 8] = gamma_dot
    return t2, trajectories2

def get_spectrum(name):
    load = True
    if load:
        z = np.load(f'trajectory_data_1_{name}.npy')
        particles = np.load(f'trajectory_data_2_{name}.npy')
    else:
        z, particles = get_trajectories(n_particles=10000, name=name)
        np.save(f'trajectory_data_1_{name}', z)
        np.save(f'trajectory_data_2_{name}', particles)
        return None
    
    particles = particles[:1000, :, :]

    t, trajectories = convert_traj(z, particles, int(round(60000 * (z[-1] / 0.6))))

    #for i in range(9):
    #    for j in range(3):
    #        if i == 2:
    #            plt.plot(t, (trajectories[j, :, i] - trajectories[j, 0, i]))
    #        else:
    #            plt.plot(t, trajectories[j, :, i])
    #    plt.savefig(f'traj_{i}.png', dpi=200)
    #    plt.clf()
        
    energies, energies_midpoint = synchrad.logspace_midpoint(3, 6, 200)
    phi_xs, phi_xs_midpoint, phi_xs_step = synchrad.linspace_midpoint(0, 7e-4, 200)
    #phi_ys, phi_ys_midpoint, phi_ys_step = synchrad.linspace_midpoint(0, 5e-4, 35)
    rad = synchrad.compute_radiation_grid(trajectories, energies, phi_xs, np.array([0.0]), np.diff(t).mean(), threads=64)
    dd = np.sum(rad ** 2, axis=3) * (0.6 / z[-1]) * (0.5e-9 / 1.602176e-19) / trajectories.shape[0]

    def plot_double_differential(double_differential, energies, energies_midpoint, thetas, thetas_midpoint, filename, name):
        fig, ax = plt.subplots()
        vmin, vmax = double_differential.min(), double_differential.max()
        ax.pcolormesh(energies_midpoint, thetas_midpoint * 1e3, double_differential.T, vmin=vmin, vmax=vmax, cmap='inferno')
        ax.set_xlim(energies.min(), energies.max())
        ax.set_ylim(thetas.min() * 1e3, thetas.max() * 1e3)
        ax.set_xscale('log')
        ax.set_xlabel('photon energy (eV)')
        ax.set_ylabel(f'$\\theta_{name}$ (mrad)')
        ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
        cbar = fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=vmin, vmax=vmax), cmap='inferno'), ax=ax)
        cbar.set_label(r'$\frac{d^2 U}{d\Omega d\epsilon}$ per particle')
        cbar.ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
        fig.savefig(filename, dpi=300)
        plt.close(fig)
        
    #plot_double_differential(dd[:, 0, :], energies, energies_midpoint, phi_ys, phi_ys_midpoint, 'ddy.png', 'y')
    plot_double_differential(dd[:, :, 0], energies, energies_midpoint, phi_xs, phi_xs_midpoint, f'ddx_symm_{name}.png', 'x')
    def blah(double_differential, phixs, phixs_mid, phiys, phiys_mid, filename):
        fig, ax = plt.subplots()
        vmin, vmax = double_differential.min(), double_differential.max()
        ax.pcolormesh(phixs_mid * 1e3, phiys_mid * 1e3, double_differential.T, vmin=vmin, vmax=vmax, cmap='inferno')
        ax.set_xlim(phixs.min() * 1e3, phixs.max() * 1e3)
        ax.set_ylim(phiys.min() * 1e3, phiys.max() * 1e3)
        ax.set_xlabel('$\\phi_x$ (mrad)')
        ax.set_ylabel('$\\phi_y$ (mrad)')
        #ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
        cbar = fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=vmin, vmax=vmax), cmap='inferno'), ax=ax)
        cbar.set_label(r'$\frac{dU}{d\Omega}$ (eV)')
        cbar.ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
        fig.savefig(filename, dpi=300)
        plt.close(fig)
        

    #angdist = np.sum(0.5 * (dd[1:, :, :] + dd[:-1, :, :]) * (energies[1:] - energies[:-1])[:, np.newaxis, np.newaxis], axis=0)
        
    #blah(angdist, phi_xs, phi_ys_midpoint, phi_ys, phi_ys_midpoint, 'ang.png')


    #spectrum = 2 * np.pi * np.sum(dd, axis=(1,2)) * phi_xs_step * phi_ys_step
    spectrum = 2 * np.pi * np.sum(dd * np.sin(phi_xs), axis=(1,2)) * phi_xs_step
    return energies, spectrum
    
energies, spectrum_m = get_spectrum('matched')
energies, spectrum_um = get_spectrum('unmatched')
fig, ax = plt.subplots()
ax.plot(energies, spectrum_m, label='matched')
ax.plot(energies, spectrum_um, label='unmatched')
plt.legend()
ax.set_xlabel('energy (eV)')
ax.set_ylabel('$\\frac{dU}{d\\epsilon}$')
ax.set_xscale('log')
ax.set_yscale('log')
fig.savefig('spectrum_symm2.png', dpi=300)
plt.close(fig)
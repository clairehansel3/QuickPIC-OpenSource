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
import os

def get_kpn1(name):
    try:
        with open(f'{name}/qpinput.json', 'r') as f:
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
    
def get_trajectories(name, n_particles='all', seed=None):

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
    kpn1 = get_kpn1(name)
    z = np.empty((len(filenames)), dtype=np.float64)
    particles = {}
    for i in range(n_particles):
        arr = np.empty((n_files, 6))
        arr[:] = np.nan
        particles[ids[i]] = arr

    # iterate through files
    for i, filename in enumerate(filenames):
        with h5.File(filename, 'r') as file:
            print(f'{name}: reading file {i+1} of {len(filenames)}')

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

def save_particles(name, n_particles, seed=None):
    z, particles = get_trajectories(name, n_particles, seed=seed)
    np.save(f'{name}/z', z)
    np.save(f'{name}/traj', particles)

def load_particles(name, n_particles):
    z = np.load(f'{name}/z.npy')
    particles = np.load(f'{name}/traj.npy')
    assert particles.shape[0] >= n_particles
    return z, particles[:n_particles, :, :]

def convert_trajectories(z, trajectories, points):
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
    
def main():
    for np in [10, 20, 30, 40, 50, 60]:
        main_x(np)

def main_x(N):
    names = ['matched', 'unmatched']
    fancy_names = ['Matched Beam', 'Mismatched Beam']
    read_hdf5_files = False
    seed = 124124#None
    n_particles_to_save = 10000
    n_particles_to_compute_with = 100
    plasma_length = 0.24
    plasma_actual_length = 0.6
    points_for_radiation_computation = 50000 * plasma_length # rule of thumb: 100,000 points per meter of plasma length
    #energies, energies_midpoint = synchrad.logspace_midpoint(3, 6.5, 20)
    #energies_is_log = True
    energies, energies_midpoint, energies_step = synchrad.linspace_midpoint(10 ** 3, 10 ** 6, N)
    energies_is_log = False
    thetas, thetas_midpoint, thetas_step = synchrad.linspace_midpoint(0, 7e-4, N)
    threads = 64
    beam_charges = [0.5e-9, 0.5e-9]
    double_differential_vmin_vmax = ['auto', 'auto']
    cmap = 'viridis'
    overal_result_suffix = '_' + str(N) + '_2D'
    
    os.system('mkdir -p results')

    if read_hdf5_files:
        for name in names:
            save_particles(name, n_particles_to_save, seed=seed)
    
    data = []    
    for i, name in enumerate(names):
        print(f'{i + 1} out of {len(names)}: computing radiation for {name}')
        z, particles = load_particles(name, n_particles_to_compute_with)
        if z[-1] < plasma_length:
            raise Exception(f'Plasma length error: cannot truncate simulation length of {z[-1]} to length {plasma_length}.')
        index = np.argmin(np.abs(z - plasma_length))
        z = z[:index]
        particles = particles[:, :index, :]
        t, trajectories = convert_trajectories(z, particles, int(round(points_for_radiation_computation)))
        rad = synchrad.compute_radiation_grid(trajectories, energies, thetas, np.array([0.0]), np.diff(t).mean(), threads=threads)
        physical_particles = beam_charges[i] / (1.60217662e-19)
        multiplier = 1
        multiplier *= physical_particles / trajectories.shape[0]
        multiplier *= plasma_actual_length / plasma_length
        data.append(np.sum(rad ** 2, axis=3) * multiplier)
        
    # plot double differential
    for i, dd in enumerate(data):
        fig, ax = plt.subplots()
        ax.set_title(fancy_names[i])
        vmin, vmax = dd.min(), dd.max()
        if double_differential_vmin_vmax[0] != 'auto':
            vmin = double_differential_vmin_vmax[0] 
        if double_differential_vmin_vmax[1] != 'auto':
            vmax = double_differential_vmin_vmax[1] 
        assert dd.shape[2] == 1
        hm = ax.pcolormesh(energies_midpoint, thetas_midpoint * 1e3, dd[:, :, 0].T, vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_xlim(energies.min(), energies.max())
        ax.set_ylim(thetas.min() * 1e3, thetas.max() * 1e3)
        if energies_is_log:
            ax.set_xscale('log')
        ax.set_xlabel('photon energy (eV)')
        ax.set_ylabel(f'$\\theta$ (mrad)')
        #ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
        cbar = fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=vmin, vmax=vmax), cmap=cmap), ax=ax)
        cbar.set_label('$\\frac{dU}{d\\Omega}$ (eV)')
        #cbar.ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
        fig.savefig(f'results/dd_{names[i]}{overal_result_suffix}.png', dpi=300)
        plt.close(fig)
        
    # plot angular dist
    fig, ax = plt.subplots()
    for i, dd in enumerate(data):
        angular_dist = np.sum(0.5 * (dd[1:, :, :] + dd[:-1, :, :]) * (energies[1:] - energies[:-1])[:, np.newaxis, np.newaxis], axis=0)
        ax.plot(thetas * 1e3, angular_dist, label=fancy_names[i])
    ax.legend()
    ax.set_xlabel('$\\theta$ (mrad)')
    ax.set_ylabel('$\\frac{dI}{d\\Omega}$')
    fig.savefig(f'results/angular_distribution{overal_result_suffix}.png', dpi=300)
    plt.close(fig)
    
    # plot spectrum
    fig, ax = plt.subplots()
    for i, dd in enumerate(data):
        spectrum = 2 * np.pi * np.sum(
            0.5 * ((dd * thetas[np.newaxis, :, np.newaxis])[:, 1:, :] + (dd * thetas[np.newaxis, :, np.newaxis])[:, :-1, :])
            * (thetas[1:] - thetas[:-1])[np.newaxis, :, np.newaxis],
            axis=(1,2))
        ax.plot(energies, spectrum, label=fancy_names[i])
    if energies_is_log:
        ax.set_xscale('log')
    xscalename = 'log' if energies_is_log else 'lin'
    ax.legend()
    ax.set_xlabel('photon energy (eV)')
    ax.set_ylabel('$\\frac{dI}{d\\epsilon}$')
    if energies_is_log:
        ax.set_yscale('log')
        yscalename = 'log'
    else:
        yscalename = 'lin'
    fig.savefig(f'results/specrum_{xscalename}_{yscalename}{overal_result_suffix}.png', dpi=300)
    #fig.savefig(f'results/specrum_{xscalename}_lin{overal_result_suffix}.png', dpi=300)
    #ax.set_yscale('log')
    #fig.savefig(f'results/specrum_{xscalename}_log{overal_result_suffix}.png', dpi=300)
    plt.close(fig)
    
    with open(f'results/total_energy{overal_result_suffix}.txt', 'w+') as f:
        f.write('simulation name\ttotal energy (eV)\n')
        for i, dd in enumerate(data):
            tot = 2 * np.pi * np.sum(0.5 * (dd[1:, :, :] + dd[:-1, :, :]) * (energies[1:] - energies[:-1])[:, np.newaxis, np.newaxis] * thetas[np.newaxis, :, np.newaxis]) * thetas_step
            f.write(f'{names[i]}\t{tot:.15e}\n')

if __name__ == '__main__':
    main()

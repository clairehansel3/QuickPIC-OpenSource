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
import matplotlib.lines as lines
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
    names = ['matched', 'unmatched']
    fancy_names = ['Matched Beam', 'Mismatched Beam']
    read_hdf5_files = False
    seed = None
    n_particles_to_save = 10000
    n_particles_to_compute_with = 100
    plasma_length = 0.24
    plasma_actual_length = 0.6
    # rule of thumb for points per meter of plasma length
    # 40,000 pretty good
    # 100,000: publication quality
    points_for_radiation_computation = 50000 * plasma_length
    energies, energies_midpoint = synchrad.logspace_midpoint(3, 6.5, 100)
    energies_is_log = True
    #energies, energies_midpoint, energies_step = synchrad.linspace_midpoint(10 ** 3, 10 ** 6, 100)
    #energies_is_log = False
    phi_xs, phi_xs_midpoint, phi_xs_step = synchrad.linspace_midpoint(-7e-4, 7e-4, 26)
    phi_ys, phi_ys_midpoint, phi_ys_step = synchrad.linspace_midpoint(-7e-4, 7e-4, 26)
    threads = 64
    beam_charges = [0.5e-9, 0.5e-9]
    double_differential_vmin_vmax = ['auto', 'auto']
    dist_vmin_vmax = ['auto', 'auto']
    cmap = 'viridis'
    overal_result_suffix = ''
    results_folder = 'results_42'
    
    
    os.system(f'mkdir -p {results_folder}')

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
        rad = synchrad.compute_radiation_grid(trajectories, energies, phi_xs, phi_ys, np.diff(t).mean(), threads=threads)
        physical_particles = beam_charges[i] / (1.60217662e-19)
        multiplier = 1
        multiplier *= physical_particles / trajectories.shape[0]
        multiplier *= plasma_actual_length / plasma_length
        data.append(np.sum(rad ** 2, axis=3) * multiplier)

    # plot double differential
    for i, dd in enumerate(data):
        fig, ax = plt.subplots()
        ax.set_title(fancy_names[i])
        dd2 = np.sum(dd, axis=2) * phi_ys_step
        vmin, vmax = dd2.min(), dd2.max()
        if double_differential_vmin_vmax[0] != 'auto':
            vmin = double_differential_vmin_vmax[0] 
        if double_differential_vmin_vmax[1] != 'auto':
            vmax = double_differential_vmin_vmax[1]
        hm = ax.pcolormesh(energies_midpoint, phi_xs_midpoint * 1e6, dd2.T, vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_xlim(energies.min(), energies.max())
        ax.set_ylim(phi_xs.min() * 1e6, phi_xs.max() * 1e6)
        if energies_is_log:
            ax.set_xscale('log')
        ax.set_xlabel('photon energy (eV)')
        ax.set_ylabel(f'$\\phi_x$ ($\\mu$rad)')
        cbar = fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=vmin, vmax=vmax), cmap=cmap), ax=ax)
        cbar.set_label('$\\frac{dI}{d\\phi_x}$ (eV)')
        fig.savefig(f'{results_folder}/dd_{names[i]}{overal_result_suffix}.png', dpi=300)
        plt.close(fig)
        
    # plot distribution
    for i, dd in enumerate(data):
            dist = np.sum(0.5 * (dd[1:, :, :] + dd[:-1, :, :]) * (energies[1:] - energies[:-1])[:, np.newaxis, np.newaxis], axis=0)
            vmin, vmax = dist.min(), dist.max()
            if dist_vmin_vmax[0] != 'auto':
                vmin = dist_vmin_vmax[0] 
            if dist_vmin_vmax[1] != 'auto':
                vmax = dist_vmin_vmax[1] 

            fig, ax = plt.subplots()
            ax.set_title(fancy_names[i])
            ax.pcolormesh(phi_xs_midpoint * 1e6, phi_ys_midpoint * 1e6, dist.T, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xlim(phi_xs.min() * 1e6, phi_xs.max() * 1e6)
            ax.set_ylim(phi_ys.min() * 1e6, phi_ys.max() * 1e6)
            ax.set_xlabel('$\\phi_x$ ($\\mu$rad)')
            ax.set_ylabel('$\\phi_y$ ($\\mu$rad)')
            cbar = fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=vmin, vmax=vmax), cmap=cmap), ax=ax)
            cbar.set_label('$\\frac{dI}{d\\Omega}$ (eV)')
            fig.savefig(f'{results_folder}/dist_{names[i]}{overal_result_suffix}.png', dpi=300)
            plt.close(fig)

    # plot angular dist
    fig, ax = plt.subplots()
    custom_lines = []
    for i, dd in enumerate(data):
        dist = np.sum(0.5 * (dd[1:, :, :] + dd[:-1, :, :]) * (energies[1:] - energies[:-1])[:, np.newaxis, np.newaxis], axis=0)
        dist_y = np.sum(dist, axis=0) * phi_xs_step
        dist_x = np.sum(dist, axis=1) * phi_ys_step
        ax.plot(phi_xs * 1e6, dist_x, color=f'C{i}', linestyle='-')
        ax.plot(phi_ys * 1e6, dist_y, color=f'C{i}', linestyle='--')
        custom_lines.append(lines.Line2D([0], [0], color=f'C{i}', lw=2, linestyle='-'))
    custom_lines.append(lines.Line2D([0], [0], color='black', lw=2, linestyle='-'))
    custom_lines.append(lines.Line2D([0], [0], color='black', lw=2, linestyle='--'))
    ax.legend(custom_lines, names + ['$\\phi_x$', '$\\phi_y$'])
    ax.set_xlabel('$\\phi$ ($\\mu$rad)')
    ax.set_ylabel('$\\frac{dI}{d\\phi}$')
    fig.savefig(f'{results_folder}/1d_dist{overal_result_suffix}.png', dpi=300)
    plt.close(fig)    

    # plot angular dist
    fig, ax = plt.subplots()
    custom_lines = []
    for i, dd in enumerate(data):
        dist = np.sum(0.5 * (dd[1:, :, :] + dd[:-1, :, :]) * (energies[1:] - energies[:-1])[:, np.newaxis, np.newaxis], axis=0)
        angles = np.linspace(0, max(max(phi_xs), max(phi_ys)), 200)
        totals = []
        for j, angle in enumerate(angles):
            mask = (phi_xs <= angle)[:, np.newaxis] & (phi_ys <= angle)[np.newaxis, :]
            totals.append(np.sum(dist[mask]) * phi_xs_step * phi_ys_step)
        ax.plot(angles * 1e6, 100 * np.array(totals) / totals[-1], label=names[i])
    ax.legend()
    ax.set_xlabel('Spectrometer Acceptance Angle ($\\mu$rad)')
    ax.set_ylabel('Percentage of Energy Deposited')
    ax.set_ylim(0, 100)
    fig.savefig(f'{results_folder}/1d_dist_alt{overal_result_suffix}.png', dpi=300)
    plt.close(fig)    
        
    # plot spectrum
    fig, ax = plt.subplots()
    for i, dd in enumerate(data):
        spectrum = np.sum(dd, axis=(1,2)) * phi_xs_step * phi_ys_step
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
    fig.savefig(f'{results_folder}/specrum_{xscalename}_{yscalename}{overal_result_suffix}.png', dpi=300)
    #fig.savefig(f'{results_folder}/specrum_{xscalename}_lin{overal_result_suffix}.png', dpi=300)
    #ax.set_yscale('log')
    #fig.savefig(f'{results_folder}/specrum_{xscalename}_log{overal_result_suffix}.png', dpi=300)
    plt.close(fig)
    
    with open(f'{results_folder}/total_energy{overal_result_suffix}.txt', 'w+') as f:
        f.write('simulation name\ttotal energy (eV)\n')
        for i, dd in enumerate(data):
            tot = np.sum(0.5 * (dd[1:, :, :] + dd[:-1, :, :]) * (energies[1:] - energies[:-1])[:, np.newaxis, np.newaxis]) * phi_xs_step * phi_ys_step
            f.write(f'{names[i]}\t{tot:.15e}\n')


if __name__ == '__main__':
    main()

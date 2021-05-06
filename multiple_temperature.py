import numpy as np
from matplotlib import pyplot as plt
from ising_montecarlo_analysis import *
import settings
import os

# esponenti critici e dimensioni
dimension = 2

alpha = 0.
beta_coeff = 1. / 8
gamma = 7. / 4
nu = 1.

beta_c = 0.4406868

# PARAMETRI per stimare errori con bootstrap migliorato
k_range = 4                                  # fino a dove calcolo il datablocking
extraction_list = list(range(250, 300, 10))  # quante fake indipendent measures prendo per bootstrap
k_tresh = 3                                  # global treshold

k_threshold_magn = k_tresh
k_threshold_C = k_tresh
k_threshold_chi = k_tresh


# preparo le figure con titoli e labels agli assi
size = 14
figsize = (6.3, 13)

fig_thermo, [ax_chi, ax_C, ax_magn] = plt.subplots(3, 1, figsize=figsize, sharex=True)

ax_chi.set_title('Suscettività magnetica, {}d'.format(dimension), size=size)
ax_chi.set_ylabel(r"$\chi$", size=size)
ax_chi.tick_params(labelsize=size)


ax_C.set_title('Calore specifico, {}d'.format(dimension), size=size)
ax_C.set_ylabel(r"$C$", size=size)
ax_C.tick_params(labelsize=size)


ax_magn.set_title('Magnetizzazione media assoluta, {}d'.format(dimension), size=size)
ax_magn.set_xlabel(r"$\beta$", size=size)
ax_magn.set_ylabel(r"$\langle|\mathcal{M}|\rangle$", size=size)
ax_magn.tick_params(labelsize=size)


fig_thermo_FSS, [ax_chi_FSS, ax_C_FSS, ax_magn_FSS] = plt.subplots(
    3, 1, figsize=figsize, sharex=True)

ax_chi_FSS.set_title('Suscettività magnetica FSS, {}d'.format(dimension), size=size)
ax_chi_FSS.set_ylabel(r"$\chi\, / \,L^{\,\gamma\,/\,\nu}$", size=size)
ax_chi_FSS.tick_params(labelsize=size)


ax_C_FSS.set_title('Calore specifico FSS, {}d'.format(dimension), size=size)
ax_C_FSS.set_ylabel(r"$C \,/\, L^{\,\alpha\,/\,\nu}$", size=size)
ax_C_FSS.tick_params(labelsize=size)


ax_magn_FSS.set_title('Magnetizzazione media assoluta FSS, {}d'.format(dimension), size=size)
ax_magn_FSS.set_xlabel(r"$(\beta-\beta_c)\,L^{\,1\,/\,\nu}$", size=size)
ax_magn_FSS.set_ylabel(r"$\langle |\mathcal{M}| \rangle \,/\, L^{\,-\beta\,/\,\nu}$", size=size)
ax_magn_FSS.tick_params(labelsize=size)

# SIMULAZIONE (PARTE 2) - ANALISI CAMBIANDO LA TEMPERATURA e le dimensioni del reticolo

# Ora guardo la magnetizzazione a diverse dimensioni
print('Dimensione:', dimension)
for lattice in [4, 8, 16, 32, 64, 128, 256]:
    print('Studiando il retico {0:d}x{0:d}...'.format(lattice))

    # carico dati da file
    simul_path = 'lattice' + str(lattice) + 'cu' + '.dat'
    with open(os.path.join(settings.DATA_DIR, simul_path), 'r') as file:
        # come prima cosa guardo la temperatura
        beta = [float(temp) for temp in next(file).split()]

        # ora guardo i dati
        mix = [[float(energ) for energ in next(file).split()]
               for temp in range(2 * len(beta))]

    mix = np.array(mix)
    energy = mix[::2]
    magn = mix[1::2]
    h = 0.0

    print('Numero di misure prese:', len(energy[0]))

    # ogni elemento delle liste è ad una temperatura corrispondente al posto beta_list
    beta_list = beta
    binder_list = []
    energ_list = []

    magn_list = []
    chi_list = []
    C_list = []

    sigma_magn_list = []
    sigma_chi_list = []
    sigma_C_list = []

    # ad ogni temperatura,
    for temp in range(len(beta)):
        # calcolo energia e magnetizzazione
        energia = np.array(energy[temp])
        magnetizzazione = np.array(np.absolute(magn[temp]))
        # e la altre quantità medie di interesse
        energy_mean, magn_mean, chi, C, binder = thermo(
            energia, magnetizzazione, dimension, lattice)
        # e appendo i risultati alle liste
        energ_list.append(energy_mean)
        binder_list.append(binder)

        magn_list.append(magn_mean)
        C_list.append(C)
        chi_list.append(chi)

        measures = len(energia)

        # poi stimo gli errori da associare a ciasuna quantità e li appendo

        # magnetizzazione stimata con autocorrel_time integrato
        sigma_magn_list.append(magnetizzazione.std() / np.sqrt(measures) *
                               np.sqrt(2 * autocorr_time_definizione(magnetizzazione)))

        # calore specifico e suscettibilità li calcolo con bootstrap migliorato
        block_size = np.arange(k_threshold_C, k_range)

        sigma_C = 0
        for k in range(k_threshold_C, k_range):
            sigma_C += sigma_bootstrap_mean(energia, extraction_list,
                                            2**k, True, lattice, dimension)
        sigma_C_list.append(sigma_C / (k_range - k_threshold_C))

        sigma_chi = 0
        for k in range(k_threshold_chi, k_range):
            sigma_chi += sigma_bootstrap_mean(magnetizzazione,
                                              extraction_list, 2**k, True, lattice, dimension)
        sigma_chi_list.append(sigma_chi / (k_range - k_threshold_chi))

    # fine analisi a temperatura variabile, ora faccio le figure

    ax_magn.errorbar(beta_list, magn_list, sigma_magn_list, label=str(
        lattice) + 'x' + str(lattice), marker='.', ls='')

    '''plt.figure("Cumulante di Binder, 2d")
                plt.scatter(beta_list, binder_list, label=str(
                    lattice) + 'x' + str(lattice), marker='.')'''

    ax_C.errorbar(beta_list, C_list, sigma_C_list, label=str(
        lattice) + 'x' + str(lattice), marker='.', ls='')

    ax_chi.errorbar(beta_list, chi_list, sigma_chi_list, label=str(
        lattice) + 'x' + str(lattice), marker='.', ls='')

    scaling_variable = (np.array(beta_list) - beta_c) * lattice ** (1 / nu)

    ax_chi_FSS.errorbar(scaling_variable, np.array(chi_list) / lattice ** (gamma / nu), np.array(sigma_chi_list) /
                        lattice ** (gamma / nu), label=str(lattice) + 'x' + str(lattice), marker='.', ls='')

    ax_C_FSS.errorbar(scaling_variable, np.array(C_list) / lattice ** (alpha / nu), np.array(sigma_C_list) /
                      lattice ** (alpha / nu), label=str(lattice) + 'x' + str(lattice), marker='.', ls='')

    ax_magn_FSS.errorbar(scaling_variable, np.array(magn_list) / lattice ** (- beta_coeff / nu), np.array(sigma_magn_list) /
                         lattice ** (- beta_coeff / nu), label=str(lattice) + 'x' + str(lattice), marker='.', ls='')


ax_chi.legend(frameon=False, fontsize='large', loc='upper left')
ax_C.legend(frameon=False, fontsize='large', loc='upper left')
ax_magn.legend(frameon=False, fontsize='large', loc='upper left')

fig_thermo.tight_layout(pad=3.0)
# fig_thermo.savefig("../figure/1/thermo_{}d.jpg".format(dimension))


'''plt.figure("Cumulante di Binder, 2d")
z = np.linspace(min(beta_list), max(beta_list), 1000)
plt.plot(z, 2. / 3 * np.ones(len(z)), color='black')
plt.xlabel('$\\beta$')
plt.ylabel('$U_4$')
plt.legend()
plt.title('binder 2D')
# plt.savefig("../figure/1/2binder.jpg")'''


# Massimo della suscettibilità va come lattice ** (gamma / nu), con gamma/nu = 7/4
# Intorno al massimo chi(beta, L) = L ** gamma/nu * phi((beta-beta_c) * L**1/nu ) relazione di scaling


ax_chi_FSS.legend(frameon=False, fontsize='large', loc='upper left')
ax_C_FSS.legend(frameon=False, fontsize='large', loc='upper left')
ax_magn_FSS.legend(frameon=False, fontsize='large', loc='upper left')

ax_chi_FSS.set_xlim([-10, 10])

fig_thermo_FSS.tight_layout(pad=3.0)
# fig_thermo_FSS.savefig("../figure/1/thermo_FSS_{}d.jpg".format(dimension))

plt.show()

import numpy as np
import settings
import os
from matplotlib import pyplot as plt
from ising_montecarlo_analysis import *

# scegliamo la simulazione da studiare
lattice = 256
dimension = 2
index_beta = 4
# index_beta = np.random.randint(0, len(beta))
save_fig = False
simul_path = 'lattice256cu.dat'

# PARAMETRI
max_correl = 200            # massima funzione di correlazione a k vicini plottata
k_range = 8                 # datablocking
bootstrap_grafico = 400     # fino a che valore del reshuffling faccio il grafico
extraction_list = list(range(250, 300, 10))  # quante fake indipendent measures prendo per bootstrap

# Datablock (on bootstrap) for mean
k_tresh = 4                 # global treshold

k_threshold_energy = k_tresh
k_threshold_magn = k_tresh
k_threshold_C = k_tresh
k_threshold_chi = k_tresh


# importiamo il file con i risultati della simulazione che ci interessa

# carico dati da file
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

# Controlliamo visivamente che sia la simulazione che ci interessa
print("Dimension = %d, lattice = %d" % (dimension, lattice))


# SIMULAZIONE (PARTE 1) - ANALISI (errori) A TEMPERATURA FISSA

# estraggo dalla lista delle temperature, energie e magnetizzazioni
# quella alla temperatura che mi interessa (specificato sopra)

my_beta = beta[index_beta]
my_energy = np.array(energy[index_beta])
my_magn = np.array(magn[index_beta])
my_magn_abs = np.absolute(my_magn)

# definisco 'measures' la lunghezza della mia markov chain (MC)
measures = len(my_energy)

# calcolo e printo energia e magn medie e deviazioni std
# NB: nel monte carlo divido per sqrt(M)
print("Lunghezza Markov Chain (numero di misure) = {0:d}".format(int(measures)))
print('Beta = {:.4f}'.format(my_beta))
print('Min number of block =', int(measures) // 2 ** k_range)
print("\nAssumendo che le misure non siano correlate (errori naive):")
print("densità energia media = {0:.5f} +/- {1:.5f},\ndensità magnetizzazione media = {2:.5f} +/- {3:.5f},\
      \ndensità magnetizzazione assoluta media = {4:.5f} +/- {5:.5f}"
      .format(my_energy.mean(), my_energy.std() / np.sqrt(measures),
              my_magn.mean(), my_magn.std() / np.sqrt(measures),
              my_magn_abs.mean(), my_magn_abs.std() / np.sqrt(measures)))

# faccio le figure di energia e magnetizzazione medie

size = 14
figsize = (6.3, 13)

x = np.arange(measures)
x_line = np.linspace(1, measures, 1000)

fig, [ax_en, ax_magn, ax_abs] = plt.subplots(3, 1, figsize=figsize, sharex=True)

ax_en.set_title('Energia media, $\\beta = ${:.4f}'.format(my_beta), size=size)
ax_en.scatter(x, my_energy, marker='.')
ax_en.plot(x_line, [my_energy.mean()] * len(x_line), color='red')
ax_en.fill_between(x_line, my_energy.mean() - my_energy.std() / np.sqrt(measures),
                   my_energy.mean() + my_energy.std() / np.sqrt(measures))
ax_en.set_ylabel(r"$\varepsilon$", size=size)
ax_en.tick_params(labelsize=size)

ax_magn.set_title('Magnetizzazione media, $\\beta = ${:.4f}'.format(my_beta), size=size)
ax_magn.scatter(x, my_magn, marker='.')
ax_magn.plot(x_line, [my_magn.mean()] * len(x_line), color='red')
ax_magn.fill_between(x_line, my_magn.mean() - my_magn.std() / np.sqrt(measures),
                     my_magn.mean() + my_magn.std() / np.sqrt(measures))
ax_magn.set_ylabel(r"$\mathcal{M}$", size=size)
ax_magn.tick_params(labelsize=size)

ax_abs.set_title('Magnetizzazione media assoluta, $\\beta = ${:.4f}'.format(my_beta), size=size)
ax_abs.scatter(x, my_magn_abs, marker='.')
ax_abs.plot(x_line, [my_magn_abs.mean()] * len(x_line), color='red')
ax_abs.fill_between(x_line, my_magn_abs.mean() - my_magn_abs.std() / np.sqrt(measures),
                    my_magn_abs.mean() + my_magn_abs.std() / np.sqrt(measures))
ax_abs.set_ylabel(r"$|\mathcal{M}|$", size=size)
ax_abs.set_xlabel(r"Steps", size=size)
ax_abs.tick_params(labelsize=size)
fig.tight_layout(pad=3.0)

if save_fig:
  fig.savefig(os.path.join(settings.FIGURE_DIR, "chains.jpg"))

# ANALISI ERRORI
if int(measures) // 2 ** k_range == 0:
  plt.show()
  raise ValueError('Il numero minumo di blocchi è uguale a zero.')

# CORRELAZIONE

figsize = (6.3, 9.5)

fig, [ax_corr, ax_int] = plt.subplots(2, 1, figsize=figsize, sharex=True)

ax_corr.set_title('Autocorrelation function', size=size)
ax_corr.scatter(np.arange(max_correl), [correlation_function(
    my_energy, k) for k in range(max_correl)], label=r'$\varepsilon$')
ax_corr.scatter(np.arange(max_correl), [correlation_function(
    my_magn_abs, k) for k in range(max_correl)], label=r'$|\mathcal{M}|$')
ax_corr.set_ylabel(r"$C(k)$", size=size)
ax_corr.tick_params(labelsize=size)
ax_corr.legend(frameon=False, fontsize='x-large')


ax_int.set_title('Autocorrelation function integrata', size=size)
ax_int.set_ylabel(r"$\Sigma_k\,\, C(k)$", size=size)
ax_int.set_xlabel(r"$k$", size=size)
ax_int.tick_params(labelsize=size)

integral_energy_sum = 0.
integral_energy_list = []
for k in range(max_correl):
  integral_energy_sum += correlation_function(my_energy, k)
  integral_energy_list.append(integral_energy_sum)
ax_int.scatter(np.arange(max_correl), integral_energy_list, label=r'$\varepsilon$', color='C0')
ax_int.hlines(y=autocorr_time_definizione(my_energy), xmin=0.,
              xmax=max_correl, color='C0', linestyle='--')

integral_magn_sum = 0.
integral_magn_list = []
for k in range(max_correl):
  integral_magn_sum += correlation_function(my_magn_abs, k)
  integral_magn_list.append(integral_magn_sum)
ax_int.scatter(np.arange(max_correl), integral_magn_list, label=r'$|\mathcal{M}|$', color='C1')
ax_int.hlines(y=autocorr_time_definizione(my_magn_abs),
              xmin=0., xmax=max_correl, color='C1', linestyle='--')

ax_int.legend(frameon=False, fontsize='x-large')

fig.tight_layout(pad=3.0)

if save_fig:
  fig.savefig(os.path.join(settings.FIGURE_DIR, "correl.jpg"))


print('\nTempi di autocorrelazione (calcolati a mano): \ntau_energia = {0:.1f}; \ntau_magn = {1:.1f}\n'.format(
    autocorr_time_definizione(my_energy), autocorr_time_definizione(my_magn_abs)))

print('sigma_e = {0:.5f}; sigma_m = {1:.5f} (naive)'.format(
    my_energy.std() / np.sqrt(measures), my_magn_abs.std() / np.sqrt(measures)))
print('sigma_e = {0:.5f}; sigma_m = {1:.5f} (tempo autocorrelazione integrato)'
      .format(my_energy.std() / np.sqrt(measures) * np.sqrt(2 * autocorr_time_definizione(my_energy)),
              my_magn_abs.std() / np.sqrt(measures) * np.sqrt(2 * autocorr_time_definizione(my_magn_abs))))

# DATABLOCKING
sigma_k_energy = []
sigma_k_magn = []

# make the chain a multiple of 2^k
for k in range(k_range):
  sigma_k_energy.append(sigma_from_blocking(my_energy, k))
  sigma_k_magn.append(sigma_from_blocking(my_magn_abs, k))

sigma_k_energy = np.asarray(sigma_k_energy)
k_scatter = np.arange(len(sigma_k_energy))
mask = k_scatter >= k_threshold_energy
sigma_db_energy = sigma_k_energy[mask].mean()

sigma_k_magn = np.asarray(sigma_k_magn)
k_scatter = np.arange(len(sigma_k_magn))
mask = k_scatter >= k_threshold_magn
sigma_db_magn = sigma_k_magn[mask].mean()


figsize = (6.3, 5)
fig, ax_db = plt.subplots(figsize=figsize)

ax_db.set_title(r'Data blocking', size=size)
ax_db.scatter(k_scatter, sigma_k_energy, label=r'$\varepsilon$', color='C0')
ax_db.hlines(y=sigma_db_energy, xmin=0., xmax=len(sigma_k_energy), color='C0', linestyle='--')

ax_db.scatter(k_scatter, sigma_k_magn, label=r'$|\mathcal{M}|$', color='C1')
ax_db.hlines(y=sigma_db_magn, xmin=0., xmax=len(sigma_k_magn), color='C1', linestyle='--')

ax_db.set_yscale("log")
ax_db.set_ylabel(r"$\sigma_k$", size=size)
ax_db.set_xlabel("$k$", size=size)
ax_db.tick_params(labelsize=size)
ax_db.legend(frameon=False, fontsize='x-large')

fig.tight_layout()

if save_fig:
  fig.savefig(os.path.join(settings.FIGURE_DIR, "data_blocking.jpg"))

print('sigma_e = {0:.5f}; sigma_m = {1:.5f} (data-blocking)'.format(sigma_db_energy, sigma_db_magn))


# BOOSTRAP (no autocorrel)
size = 14

# 100 indipendent samples mi danno 10% accuracy
# 200 indipendent samples mi danno 5% accuracy

sigma_j_bootstrap_chi = []
sigma_j_bootstrap_C = []

for j in range(2, bootstrap_grafico):
  sigma_j_bootstrap_C.append(sigma_from_bootstrap_blocking(
      my_energy, j, block_size=1, dispersion=True, lattice=lattice, dimension=dimension))
  sigma_j_bootstrap_chi.append(sigma_from_bootstrap_blocking(
      my_magn_abs, j, block_size=1, dispersion=True, lattice=lattice, dimension=dimension))


figsize = (6.3, 5)
fig, ax_bs = plt.subplots(figsize=figsize)

ax_bs.set_title("Bootstrap", size=size)
ax_bs.scatter(np.arange(2, bootstrap_grafico), sigma_j_bootstrap_C, label=r'$C$', color='C2')
ax_bs.scatter(np.arange(2, bootstrap_grafico), sigma_j_bootstrap_chi, label=r'$\chi$', color='C3')

boot_mean_energy = sigma_bootstrap_mean(my_energy, extraction_list, 1, False, lattice, dimension)
boot_mean_magn = sigma_bootstrap_mean(my_magn_abs, extraction_list, 1, False, lattice, dimension)

print('sigma_e = {0:.5f}; sigma_m = {1:.5f} (bootstrap, no correl)'.format(
    boot_mean_energy, boot_mean_magn))

energy_mean, magn_mean, chi, C, binder = thermo(my_energy, my_magn, dimension, lattice)


sigma_C = sigma_bootstrap_mean(my_energy, extraction_list, 1, True, lattice, dimension)
sigma_chi = sigma_bootstrap_mean(my_magn_abs, extraction_list, 1, True, lattice, dimension)


print('\nCalore specifico = {0:.5f} +/- {1:.5f}; Chi = {2:.5f} +/- {3:.5f} (bootstrap, no correl)'.format(
    C, sigma_C, chi, sigma_chi))

ax_bs.hlines(y=sigma_C, xmin=0., xmax=len(sigma_j_bootstrap_C), color='C2', linestyle='--')
ax_bs.hlines(y=sigma_chi, xmin=0., xmax=len(sigma_j_bootstrap_chi), color='C3', linestyle='--')


ax_bs.set_ylabel(r"$\sigma_j$", size=size)
ax_bs.set_xlabel("$j$", size=size)
ax_bs.legend(frameon=False, fontsize='x-large', loc='lower right')
ax_bs.tick_params(labelsize=size)

fig.tight_layout()

if save_fig:
  fig.savefig(os.path.join(settings.FIGURE_DIR, "bootstrap.jpg"))


# BOOSTRAP MIGLIORATO

# Energy and magn
block_size = np.arange(k_range)
sigma_k_bootstrap_migliorato_energy = []
sigma_k_bootstrap_migliorato_magn = []

for k in block_size:
  sigma_k_bootstrap_migliorato_energy.append(sigma_bootstrap_mean(
      my_energy, extraction_list, 2**k, False, lattice, dimension))
  sigma_k_bootstrap_migliorato_magn.append(sigma_bootstrap_mean(
      my_magn_abs, extraction_list, 2**k, False, lattice, dimension))

sigma_k_bootstrap_migliorato_energy = np.asarray(sigma_k_bootstrap_migliorato_energy)
sigma_k_bootstrap_migliorato_magn = np.asarray(sigma_k_bootstrap_migliorato_magn)


figsize = (6.3, 5)

fig, ax_bsm = plt.subplots(figsize=figsize)

ax_bsm.set_title("Bootstrap migliorato", size=size)

ax_bsm.set_ylabel(r"$\sigma_k$", size=size)
ax_bsm.set_xlabel("$k$", size=size)
ax_bsm.legend(frameon=False, fontsize='x-large')
ax_bsm.tick_params(labelsize=size)
ax_bsm.set_yscale('log')
# plt.ylim(min(sigma_k_bootstrap_migliorato_energy) / 2.,
#         2 * max(sigma_k_bootstrap_migliorato_energy))

# Energy
block_size_scatter = np.arange(len(sigma_k_bootstrap_migliorato_energy))
mask = block_size_scatter >= k_threshold_energy

sigma_energy = sigma_k_bootstrap_migliorato_energy[mask].mean()
ax_bsm.scatter(block_size_scatter, sigma_k_bootstrap_migliorato_energy,
               label=r'$\varepsilon$', color='C0')
ax_bsm.hlines(y=sigma_energy, xmin=0., xmax=len(sigma_k_bootstrap_migliorato_energy),
              color='C0', linestyle='--')

# Magnetization
block_size_scatter = np.arange(len(sigma_k_bootstrap_migliorato_magn))
mask = block_size_scatter >= k_threshold_magn

sigma_magn = sigma_k_bootstrap_migliorato_magn[mask].mean()
ax_bsm.scatter(block_size_scatter, sigma_k_bootstrap_migliorato_magn,
               label=r'$|\mathcal{M}|$', color='C1')
ax_bsm.hlines(y=sigma_magn, xmin=0., xmax=len(
    sigma_k_bootstrap_migliorato_magn), color='C1', linestyle='--')


# C and chi
sigma_k_bootstrap_migliorato_C = []
sigma_k_bootstrap_migliorato_chi = []

for k in range(k_range):
  sigma_k_bootstrap_migliorato_C.append(
      sigma_bootstrap_mean(my_energy, extraction_list, 2**k, True, lattice, dimension))
  sigma_k_bootstrap_migliorato_chi.append(
      sigma_bootstrap_mean(my_magn_abs, extraction_list, 2**k, True, lattice, dimension))

sigma_k_bootstrap_migliorato_C = np.asarray(sigma_k_bootstrap_migliorato_C)
sigma_k_bootstrap_migliorato_chi = np.asarray(sigma_k_bootstrap_migliorato_chi)

block_size_scatter = np.arange(len(sigma_k_bootstrap_migliorato_C))

mask = block_size_scatter >= k_threshold_C
sigma_C = sigma_k_bootstrap_migliorato_C[mask].mean()


ax_bsm.scatter(block_size_scatter, sigma_k_bootstrap_migliorato_C,
               label=r'$C$', color='C2')
ax_bsm.hlines(y=sigma_C, xmin=0., xmax=len(sigma_k_bootstrap_migliorato_C),
              color='C2', linestyle='--')

block_size_scatter = np.arange(len(sigma_k_bootstrap_migliorato_chi))

mask = block_size_scatter >= k_threshold_chi
sigma_chi = sigma_k_bootstrap_migliorato_chi[mask].mean()

ax_bsm.scatter(block_size_scatter, sigma_k_bootstrap_migliorato_chi,
               label=r'$\chi$', color='C3')
ax_bsm.hlines(y=sigma_chi, xmin=0., xmax=len(sigma_k_bootstrap_migliorato_chi),
              color='C3', linestyle='--')

ax_bsm.legend(frameon=False, fontsize='x-large', loc='lower right', ncol=2)
fig.tight_layout()

if save_fig:
  fig.savefig(os.path.join(settings.FIGURE_DIR, "bootstrap_migliorato.jpg"))


print('Calore specifico = {0:.5f} +/- {1:.4f}; Chi = {2:.5f} +/- {3:.5f} (bootstrap migliorato)'.format(
    C, sigma_C, chi, sigma_chi))

# per le variabili iniziali
print('\nsigma_e = {0:.5f}; sigma_m = {1:.5f} (bootstrap migliorato)'.format(
    sigma_energy, sigma_magn))

plt.show()

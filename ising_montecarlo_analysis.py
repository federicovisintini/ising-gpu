import numpy as np
from numba import jit
import random


@jit
def thermo(energy, magnetization, dimension, lattice):
    '''
    energy, magnetization: sono due array contenenti l'energia e la
                           magnetizzazione in una data simulazione
    return: una tupla di grandezze termodinamiche interessanti, i.e.
        - energia media
        - parametro d'ordine (i.e. media di |M|)
        - lunghezza di correlazione (NON IMPLEMENTATA)
        - suscettività
        - calore specifico
        - cumulante di binder
    '''

    # energia media
    energy_mean = energy.mean()
    # magnetizzazione media, con valore assoluto
    magn_mean = np.abs(magnetization).mean()
    # suscettività magnetica
    chi = lattice**dimension * (np.mean(magnetization**2) - magn_mean**2)
    # calore specifico
    C = lattice**dimension * (np.mean(energy**2) - energy_mean**2)
    # cumulante di binder
    binder = 1. - (np.mean(magnetization**4)) / (3. * (np.mean(magnetization**2))**2)

    return energy_mean, magn_mean, chi, C, binder


@jit
def correlation_function(chain, i=0):
    ''' mi calcolo la funzione di autocorrelazione i-esima per la media
        della variabile casuale campionata dalla chain (MCMC), calcolata a mano
        attraverso la formula diretta: <O[k]O[k+i]> '''

    correlation_func = 0.0

    chain_mean = chain.mean()
    chain_var = np.mean(chain**2)

    # sommo su tutta la MC la correlazione fra i e i+k. Poi divido per fare la media
    for k in range(len(chain) - i):
        correlation_func += (chain[k] - chain_mean) * (chain[k + i] - chain_mean)

    # faccio la media
    return correlation_func / (len(chain) - i) / (chain_var - chain_mean ** 2)


@jit
def autocorr_time_definizione(chain):
    ''' dalla definizione di autocorrelazione, ho una MC, valuto <O_k * O_k+i>,
        valuto gli estremi di integrazione con window function per minimizzare l'errore
        se ci mette tanto si può valutare l'idea di usare FFT
        la cosa migliore non sarebbe stimarlo direttamente, ma data blocking / resampling '''

    N = len(chain)
    tau1 = 0

    # dovrei integrare fino a infinito, ma c'è rumore integro fino ad plateau (leggere articoli)
    # introduco la summation window 'W', e sbaglio la somma di exp(-W/tau)
    # scelgo W in modo da minimizzare l'errore totale (sistematico + rumore): exp(-W/tau)+sqrt(W/N)

    # faccio quindi una prima stima grezzissima di tau, alla prima corr_function che scende sotto var(sigma)/e
    for i in range(1, N):
        if correlation_function(chain, i) < 0.368:
            tau1 = i
            break

    # mi calcolo approssimativamente la mia window function W, minimizzando exp(-W/tau)+sqrt(W/N)
    # mi calcolo la derivata: exp(-W/tau) - tau/sqrt(W*N), il primo W che rende negativa l'espressione è quello scelto:
    # prendo come primo valore del plateau uno che non mi da problemi se tau è grandino
    W = 1 + int(2 * tau1**2 / N)
    while((np.exp(-W / tau1) - tau1 / np.sqrt(W * N)) > 0):
        W += 1

    tau2 = 0.  # azzero il valore di tau
    for i in range(W):
        # intergro la funzione di correlazione i-esima per ogni i, fino al plateau
        tau2 += correlation_function(chain, i)

    return tau2


@jit
def sigma_from_blocking(chain, k):
    ''' ritorna la stima per la deviazione standard usando un DATA BLOCKING
        con blocchi di 2^k dati, funziona per energy / magn con autocorrel '''

    number_of_blocks = len(chain) // 2 ** k
    if number_of_blocks == 0:
        print('number of blocks = 0')
        return 1

    blocks = np.zeros(number_of_blocks)

    for block in range(number_of_blocks):
        for j in range(2 ** k):
            blocks[block] += chain[block * 2 ** k + j]
        blocks[block] /= 2**k

    return np.std(blocks) / np.sqrt(len(blocks))


@jit
def sigma_from_bootstrap_blocking(chain, number_fake_extraction, block_size=1, dispersion=True, lattice=10, dimension=2):
    ''' moralmente faccio data blocking su bootstrap, questa è la funzione che useremo sempre
        ponendo chain = energy / magnetization:
            se dispersion == False calcolo sigma energia / magnetization
            se dispersion == True calcolo sigma calore specifico / suscettibilità '''

    # inizializzo un array 'copies' dove salvare i risultati dele simulazioni
    copies = np.zeros(number_fake_extraction)
    number_of_blocks = len(chain) // block_size
    len_chain_stripped = block_size * number_of_blocks
    if number_of_blocks == 0:
        print('number of blocks = 0')
        return 0

    if dispersion is False:
        for extraction in range(number_fake_extraction):
            fake_observable = 0.0
            for block in range(number_of_blocks):
                rand_index = random.randrange(len(chain))
                for index in range(block_size):
                    # F_N è la media
                    fake_observable += chain[rand_index - index]
            copies[extraction] = fake_observable / len_chain_stripped

    elif dispersion is True:
        chain_2 = chain ** 2  # cerco di ottimizzare
        for extraction in range(number_fake_extraction):
            fake_observable = 0.0
            fake_observable_2 = 0.0
            for block in range(number_of_blocks):
                rand_index = random.randrange(len(chain))
                for index in range(block_size):
                    fake_observable += chain[rand_index - index]
                    fake_observable_2 += chain_2[rand_index - index]

            copies[extraction] = fake_observable_2 / len_chain_stripped - \
                (fake_observable / len_chain_stripped) ** 2

        copies = copies * lattice ** dimension

    return np.std(copies)


def sigma_bootstrap_mean(chain, extractions_list=list(range(250, 300, 10)), block_size=1, dispersion=True, lattice=10, dimension=2):

    boot_mean = 0

    for extraction in extractions_list:
        boot_mean += sigma_from_bootstrap_blocking(
            chain, extraction, block_size, dispersion, lattice, dimension)

    return boot_mean / len(extractions_list)


'''
# fede implementation for binder

binder = 1. * np.arange(j)
for i in range(j):
    tmp2 = 0
    tmp4 = 0
    for k in range(len(chain)):
        tmp = chain[np.random.randint(0, len(chain))]
        tmp2 += tmp ** 2
        tmp4 += tmp ** 4
    binder[i] = tmp4 / (3 * tmp2)
return np.sqrt(np.mean(binder**2) - np.mean(binder)**2)

# elia implementation for binder

extractions = []
for i in range(j * len(chain)):
    random_index = np.random.randint(0, len(chain))
    extractions.append(chain[random_index])
resamples = np.asarray(extractions)
resamples_reshaped = resamples.reshape((j, len(chain)))
binder = np.mean(resamples_reshaped**4, axis=1) / 3 / \
    np.mean(resamples_reshaped**2, axis=1)**2
return binder.std()
'''

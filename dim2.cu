#include <stdio.h>
#include <stdlib.h>

#include <time.h>
#include <math.h>

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#define filename "data/lattice256cu.dat"

#define lattice 256                     // grandezza reticolo
#define num_beta 20                     // # di temperature visitate
#define num_misure 9000                 // # di volte che si chiede energy e magnetization
#define num_metropolis_fra_misure 1     // # aggiornamenti matrice prima di chiedere energia e magn
#define beta_c 0.44                     // beta critico in dimensione 2
#define termalizzazione 100
#define magn_field 0.0

#define THREADs 1024
#define BLOCKs 128

// salva il risultato della simulazione nel file 'lattice<lattice>cu.dat' nella forma:

// lista beta
// lista energie campionate per beta[0]
// lista magn campionate per beta[0]
// lista energie campionate per beta[1]
// lista magn campionate per beta[1]
// ...
// ...
// lista energie campionate per beta[num_temp]
// lista magn campionate per beta[num_temp]

// per un totale di 2 * num_beta + 1 righe
// in ogni riga gli elementi sono separati da spazi

#define CHECK_CUDA(call) {                                                   \
    cudaError_t err = call;                                                  \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
    }}

#define CHECK_CURAND(call) {                                                 \
    curandStatus_t status = call;                                            \
    if( CURAND_STATUS_SUCCESS != status) {                                   \
        fprintf(stderr, "CURAND error: %s = %d at (%s:%d)\n", #call,         \
                status, __FILE__, __LINE__);                                 \
        exit(EXIT_FAILURE);                                                  \
    }}

// dichiarazione funzioni
void presa_misure(short *p, short * dev_p, float * beta, float * energ, float * magn, float * rand_vals, curandGenerator_t rng);
__global__ void step_metro_scacchiera(short *p, float beta, float *rand_val, short color);
float energy(short * p);
float magnetization(short * p);

int main(){
    // INITIALIZE VARIABLES

    // Initialise variable for HOST
    // le matrici le stiro ad array 1D
    float random;
    int i, j;

    short *p;
    float beta[num_beta], *energ, *magn;

    // Initialise variable for DEVICE
    // le matrici le stiro ad array 1D
    short *dev_p;
    float *dev_beta; // *dev_energ, *dev_magn;

    // SETUP cuRAND generator // XORWOW generator ??
    curandGenerator_t rng;
    float *rand_vals;
    CHECK_CURAND(curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_PHILOX4_32_10));
    CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(rng, time(NULL)));
    CHECK_CUDA(cudaMalloc(&rand_vals, lattice * lattice * sizeof(*rand_vals)));

    // and normal random generator
    srand(time(NULL));

    // ALLOCATE MEMORY

    // Allocate memory on HOST
    // creo matrice degli spin in modo dinamico
    p = (short *)malloc(lattice * lattice * sizeof(short));

    // alloco lo spazio per le matrici energia e magn
    energ = (float *)malloc(num_beta * num_misure * sizeof(float));
    magn = (float *)malloc(num_beta * num_misure * sizeof(float));

    // Allocate memory on DEVICE
    CHECK_CUDA(cudaMalloc(&dev_p, lattice * lattice * sizeof(short)));
    CHECK_CUDA(cudaMalloc(&dev_beta, sizeof(float) * num_beta));
    // CHECK_CUDA(cudaMalloc(&dev_energ, num_misure * num_beta * sizeof(float)));
    // CHECK_CUDA(cudaMalloc(&dev_magn, num_misure * num_beta * sizeof(float)));


    // riempio l'array delle temperature
    for(i=0; i<num_beta; i++){
        beta[i] = -0.66 + 2 * 0.66 * i / num_beta;
        beta[i] = beta[i] * beta[i] * beta[i] + beta[i] / 10 + beta_c;
    }

    // inizializzo la matrice "p" 'a caldo': oriento casualmente gli spin
    for(int x=0; x<lattice; x++){
        for(int y=0; y<lattice; y++){
            random = rand()%2;
            if (random == 0) p[x * lattice + y] = 1;
            else p[x * lattice + y] = -1;
        }
    }

    // TRANSFER DATA FROM HOST TO DEVICE MEMORY
    CHECK_CUDA(cudaMemcpy(dev_p, p, sizeof(short) * lattice * lattice, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dev_beta, beta, sizeof(float) * num_beta, cudaMemcpyHostToDevice));

    // facciamo il metropolis
    presa_misure(p, dev_p, beta, energ, magn, rand_vals, rng);

    // TRANSFER DATA FROM DEVICE TO HOST MEMORY
    // CHECK_CUDA(cudaMemcpy(energ, dev_energ, sizeof(float) * num_misure * num_beta, cudaMemcpyDeviceToHost));
    // CHECK_CUDA(cudaMemcpy(magn, dev_magn, sizeof(float) * num_misure * num_beta, cudaMemcpyDeviceToHost));

    // salviamo il risultato nella forma sopra specificata
    FILE *file = fopen(filename, "w");
    // lista dei beta
    for(i=0; i<num_beta; i++) fprintf(file, "%f ", beta[i]);
    // matrice energ e magn
    for(i=0; i<num_beta; i++){
        fprintf(file, "\n");
        for(j=0; j<num_misure; j++) fprintf(file, "%f ", energ[i*num_misure + j]);
        fprintf(file, "\n");
        for(j=0; j<num_misure; j++) fprintf(file, "%f ", magn[i*num_misure + j]);
    }
    fclose(file);

    // free memory on device
    CHECK_CUDA(cudaFree(dev_p));
    CHECK_CUDA(cudaFree(dev_beta));
    CHECK_CUDA(cudaFree(rand_vals));
    // CHECK_CUDA(cudaFree(dev_energ));
    // CHECK_CUDA(cudaFree(dev_magn));

    // libero lo spazio delle matrici energ e magn
    free(energ); free(magn); free(p);
    return 0;
}



void presa_misure(short *p, short * dev_p, float * beta, float * energ, float * magn, float * rand_vals, curandGenerator_t rng){
    // funzione vera e propria, esegue l'algoritmo salvando i risultati in energ e magn

    // Setup CUDA launch configuration
    int blocks = (lattice * lattice + THREADs - 1) / THREADs;

    // variando le temperature
    for(int i=0; i<num_beta; i++){
        // warm-up (termalizzazione)
        for(int j=0; j<termalizzazione; j++){

            CHECK_CURAND(curandGenerateUniform(rng, rand_vals, lattice*lattice));
            step_metro_scacchiera<<<blocks, THREADs>>>(dev_p, beta[i], rand_vals, 0);

            CHECK_CURAND(curandGenerateUniform(rng, rand_vals, lattice*lattice));
            step_metro_scacchiera<<<blocks, THREADs>>>(dev_p, beta[i], rand_vals, 1);
            }
        // chiedo energia e magnetizzazione M volte
        for(int j=0; j<num_misure; j++){
            // facendo N chiamate dell'intero reticolo
            for(int k=0; k<num_metropolis_fra_misure; k++){

                CHECK_CURAND(curandGenerateUniform(rng, rand_vals, lattice*lattice));
                step_metro_scacchiera<<<blocks, THREADs>>>(dev_p, beta[i], rand_vals, 0);

                CHECK_CURAND(curandGenerateUniform(rng, rand_vals, lattice*lattice));
                step_metro_scacchiera<<<blocks, THREADs>>>(dev_p, beta[i], rand_vals, 1);
            }
            // ora mi calcolo l'energia e la magnetizzazione
            CHECK_CUDA(cudaMemcpy(p, dev_p, sizeof(short) * lattice * lattice, cudaMemcpyDeviceToHost));
            energ[i*num_misure + j] = energy(p);
            magn[i*num_misure + j] = magnetization(p);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
    }
}



__global__ void step_metro_scacchiera(short *dev_p, float beta, float *rand_vals, short color){
    // Esegue l'algoritmo di metropolis su tutto il reticolo andando a "scacchiera"
    // prima si girano tutti i p.ti della forma (2a, 2b) e (2a+1, 2b+1) con 'a' e 'b' interi
    // poi tutti quella della forma (2a+1, 2b) e (2a+1, 2b).
    // L'idea è che giriamo tutti gli spin e poi aggiorniamo la matrice duale
    // e come vengono girati gli spin non dipende da come sono stati girati quelli prima
    // ma dopo l'ultima chiamata dell'aggiornamento della matrice duale.
    // Poi parallerizzeremo il conto di ognuna delle 2 sottoscacchiere (bianchi e neri)

    const long long id = static_cast<long long>(blockDim.x) * blockIdx.x + threadIdx.x;
    const int i = id / lattice;
    const int j = id % lattice;
    if (i >= lattice || j >= lattice) return;
    if ((i+j) % 2 != color) return;

    int ip, in, jp, jn;
    ip = i+1;
    in = i-1;
    jp = j+1;
    jn = j-1;

    if(ip==lattice) ip = 0;
    if(in==-1) in = lattice-1;
    if(jp==lattice) jp = 0;
    if(jn==-1) jn = lattice-1;

    if (rand_vals[id] < exp(-2 * dev_p[i*lattice + j] * (dev_p[in*lattice + j] + dev_p[ip*lattice + j] + dev_p[i*lattice + jn] + dev_p[i*lattice + jp] + magn_field) * beta))  // accettanza del metropolis
        dev_p[id] = - dev_p[id];  // se accetto faccio spin flip

}


float energy(short * p){
    // calcola l'energia di una certa configurazione p

    float energy = 0;
    // calcolo energia per elementi della matrice e li sommo

    int ip, in, jp, jn;
    for(int x=0; x<lattice; x++){
        for(int y=0; y<lattice; y++){
            ip = x+1;
            in = x-1;
            jp = y+1;
            jn = y-1;

            if(ip==lattice) ip = 0;
            if(in==-1) in = lattice-1;
            if(jp==lattice) jp = 0;
            if(jn==-1) jn = lattice-1;

            energy -= (p[in*lattice + y] + p[ip*lattice + y] + p[x*lattice + jn] + p[x*lattice + jp] + 2 * magn_field) * p[x*lattice + y] / 2.;
        }
    }
    return energy / (lattice * lattice);
}


float magnetization(short * p){
    // calcola la magnetizzazione di una certa configurazione p

    float magn = 0;
    for(int x=0; x<lattice; x++){
        for(int y=0; y<lattice; y++)
            magn += p[x*lattice+y];
    }
    return magn / (lattice * lattice);
}

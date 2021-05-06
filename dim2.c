#include <stdio.h>
#include <stdlib.h>

#include <time.h>
#include <math.h>

#define filename "data/lattice256c.dat"

#define lattice 256                     // grandezza reticolo
#define num_beta 20                     // # di temperature visitate
#define num_misure 9000                 // # di volte che si chiede energy e magnetization
#define num_metropolis_fra_misure 1     // # aggiornamenti matrice prima di chiedere energia e magn
#define beta_c 0.44                     // beta critico in dimensione 2

// salva il risultato della simulazione nel file '/DATA/lattice<lattice>_c.dat' nella forma:

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

// dichiarazione funzioni
float energy(short ** p, short ** S, float h);
float magnetization(short ** p);
void presa_misure(short **p, short **S, float * beta, float h, float ** energ, float ** magn, float *rand_val);
void step_metro_scacchiera(short **p, short **S, float beta, float h, float *rand_val);
void matrice_duale(short **p, short ** S);

// funzione che riempie un array lattice * lattice di numeri generati casualmente
void my_rand(float *rand_val){
    for(int rand_index=0; rand_index<lattice*lattice; rand_index++)
        rand_val[rand_index] = rand() / (RAND_MAX + 1.0);
    return;
}

int main(){
    srand(time(NULL));
    srand(404); // for developing
    short **p, **S;
    int i;
    float *rand_val;
    float random, prob, x, h = 0.0;
    float beta[num_beta], **energ, **magn;

    // creo matrice e duale, purtroppo da generare in modo dinamico
    p = (short **)malloc(lattice * sizeof(short *));
    for (i = 0; i < lattice; i++) p[i] = (short *)malloc(lattice * sizeof(short));
    S = (short **)malloc(lattice * sizeof(short *));
    for (i = 0; i < lattice; i++) S[i] = (short *)malloc(lattice * sizeof(short));

    // alloco lo spazio per le matrici energia e magn
    energ = (float **)malloc(num_beta * sizeof(float *));
    for (i = 0; i < num_beta; i++) energ[i] = (float *)malloc(num_misure * sizeof(float));
    magn = (float **)malloc(num_beta * sizeof(float *));
    for (i = 0; i < num_beta; i++) magn[i] = (float *)malloc(num_misure * sizeof(float));

    // alloco lo spazio per la matrice di numeri casuali
    rand_val = (float *)malloc(lattice * lattice * sizeof(float));

    // riempio l'array delle temperature in modo gaussiano intorno a beta_c
    for (i=0; i < num_beta; i++){
        x = rand() / (RAND_MAX + 1.0); // provo a usare questo beta
        prob = exp((beta_c - x)*(beta_c - x) / 0.05); // probabilità di accettare x
        random = rand() / (RAND_MAX + 1.0); // tiro moneta per vedere se accetto x
        if (random < prob) beta[i] = x; // se accetto x
        else i--;                       // altrimenti ritento
    }
    //beta[0] = 0.44; // testing

    // ordino beta (serve davvero?)
    // MISSING

    // inizializzo la matrice "p" 'a caldo': oriento casualmente gli spin
    for(int x=0; x<lattice; x++){
        for(int y=0; y<lattice; y++){
            if (rand()%2 == 0) p[x][y] = 1;
            else p[x][y] = -1;
        }
    }

    // facciamo il metropolis
    presa_misure(p, S, beta, h, energ, magn, rand_val);

    // libero lo spazio della matrice e della duale
    for (i = 0; i < lattice; i++) free(p[i]);
    for (i = 0; i < lattice; i++) free(S[i]);
    free(p); free(S);
    free(rand_val);

    // salviamo il risultato nella forma sopra specificata
    FILE *file = fopen(filename, "w");
    // lista dei beta
    for(i=0; i<num_beta; i++) fprintf(file, "%f ", beta[i]);
    // matrice energ e magn
    for(i=0; i<num_beta; i++){
        fprintf(file, "\n");
        for(int j=0; j<num_misure; j++) fprintf(file, "%f ", energ[i][j]);
        fprintf(file, "\n");
        for(int j=0; j<num_misure; j++) fprintf(file, "%f ", magn[i][j]);
    }
    fclose(file);

    // libero lo spazio delle matrici energ e magn
    for (i = 0; i < num_beta; i++) free(energ[i]);
    for (i = 0; i < num_beta; i++) free(magn[i]);
    free(energ); free(magn);
    return 0;
}


void presa_misure(short **p, short **S, float * beta, float h, float ** energ, float ** magn, float * rand_val){
    // funzione vera e propria, esegue l'algoritmo salvando i risultati in energ e magn

    // mi calcolo la matrice duale date le cond iniziali
    matrice_duale(p, S);

    // variando le temperature
    for(int i=0; i<num_beta; i++){
        // chiedo energia e magnetizzazione M volte
        for(int j=0; j<num_misure; j++){
            // facendo N chiamate dell'intero reticolo
            for(int k=0; k<num_metropolis_fra_misure; k++){
                my_rand(rand_val);
                step_metro_scacchiera(p, S, beta[i], h, rand_val);
            }
            energ[i][j] = energy(p, S, h);
            magn[i][j] = magnetization(p);
        }
    }
}


float energy(short ** p, short ** S, float h){
    // calcola l'energia di una certa configurazione p
    // usando una matrice di appoggio S

    float energy = 0;
    // calcolo energia per elementi della matrice e li sommo
    for(int x=0; x<lattice; x++){
        for(int y=0; y<lattice; y++)
            energy -= (S[x][y] + 2 * h) * p[x][y] / 2.;
    }
    return energy / (lattice * lattice);
}


float magnetization(short ** p){
    // calcola la magnetizzazione di una certa configurazione p

    float magn = 0;
    for(int x=0; x<lattice; x++){
        for(int y=0; y<lattice; y++)
            magn += p[x][y];
    }
    return magn / (lattice * lattice);
}


void step_metro_scacchiera(short **p, short **S, float beta, float h, float *rand_val){
    // Esegue l'algoritmo di metropolis su tutto il reticolo andando a "scacchiera"
    // prima si girano tutti i p.ti della forma (2a, 2b) e (2a+1, 2b+1) con 'a' e 'b' interi
    // poi tutti quella della forma (2a+1, 2b) e (2a+1, 2b).
    // L'idea è che giriamo tutti gli spin e poi aggiorniamo la matrice duale
    // e come vengono girati gli spin non dipende da come sono stati girati quelli prima
    // ma dopo l'ultima chiamata dell'aggiornamento della matrice duale.
    // Poi parallerizzeremo il conto di ognuna delle 2 sottoscacchiere (bianchi e neri)

    int rand_index=0;
    float prob, random;

    for(int x=0; x<lattice; x++){
        for(int y=0; y<lattice; y++){
            if((x+y) % 2 == 0){
                prob = exp(-2 * p[x][y] * (S[x][y] + h) * beta);
                random = rand_val[rand_index]; // (RAND_MAX + 1.0);
                rand_index ++;
                if (random < prob)  // accettanza del metropolis
                    p[x][y] = - p[x][y];  // se accetto faccio spin flip
            }
        }
    }

    matrice_duale(p, S);

    for(int x=0; x<lattice; x++){
        for(int y=0; y<lattice; y++){
            if((x+y) % 2 == 1){
                prob = exp(-2 * p[x][y] * (S[x][y] + h) * beta);
                random = rand_val[rand_index]; // (RAND_MAX + 1.0);
                rand_index ++;
                if (random < prob)  // accettanza del metropolis
                    p[x][y] = - p[x][y];  // se accetto faccio spin flip
            }
        }
    }
    matrice_duale(p, S);
}


void matrice_duale(short **p, short ** S){
    // aggiorna la sottoscacchiera S, riempiendola con gli elementi della "matrice duale"
    // è un po' lungo, ma così è più veloce che definire altre variabili
    int ip, in, jp, jn;
    // calcolo energia per elementi centrali della matrice
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

            S[x][y] = p[in][y] + p[ip][y] + p[x][jn] + p[x][jp];
        }
    }
}

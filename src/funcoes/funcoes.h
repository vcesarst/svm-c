#ifndef FUNCOES_H
#define FUNCOES_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

// Estrutura que encapsula a matematica do nosso SVM
typedef struct {
    double *alpha;   // Multiplicadores de Lagrange (tamanho N)
    double *w;       // Vetor normal geometrico (tamanho d)
    double b;        // Vies / Intercepto
    int N;           // Quantidade de amostras de treino
    int d;           // Numero de dimensoes (features)
    double C;        // Parametro de penalidade (Soft Margin)
    double tol;      // Tolerancia para violacao KKT
    int max_passes;  // Criterio de parada do SMO
} SVM_Model;

// Declaracao das funcoes de ciclo de vida e IO
int ler_dataset(const char *nome_arquivo, double **X, double **y, int *N, int d);
SVM_Model* alocar_svm(int N, int d, double C, double tol, int max_passes);
void liberar_svm(SVM_Model *modelo);

// Declaracao das funcoes do motor matematico
double kernel_linear(const double *X, int i, int j, int d);
void treinar_smo(SVM_Model *modelo, const double *X, const double *y);
double prever(SVM_Model *modelo, const double *x_teste);

#endif

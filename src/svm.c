#include <stdio.h>
#include <stdlib.h>
#include "funcoes/funcoes.h"

int main() {
    int N = 0; 
    int d = 4; // Iris tem 4 dimensoes
    
    double *X = NULL;
    double *y = NULL;
    
    // Caminho relativo mapeado de src/ para examples/
    const char *caminho_arquivo = "../examples/iris_scaled.txt";
    
    printf("=========================================\n");
    printf("   TREINAMENTO SVM (SMO) EM C - IRIS\n");
    printf("=========================================\n");
    
    if (!ler_dataset(caminho_arquivo, &X, &y, &N, d)) {
        printf("Falha ao carregar o dataset. Verifique o caminho.\n");
        return 1;
    }
    
    printf("[OK] Dataset carregado: %d amostras, %d features.\n", N, d);
    
    // Hiperparametros (Alinhe o C com o parametro usado no seu Python)
    double C = 1.0;         
    double tol = 0.001;     
    int max_passes = 20;    
    
    SVM_Model *modelo = alocar_svm(N, d, C, tol, max_passes);
    
    printf("[..] Iniciando otimizacao de Karush-Kuhn-Tucker...\n");
    treinar_smo(modelo, X, y);
    
    printf("\n=== VETORES EXTRAIDOS ===\n");
    for (int k = 0; k < d; k++) {
        printf("w[%d] = %f\n", k, modelo->w[k]);
    }
    printf("b    = %f\n", modelo->b);
    
    int sv_count = 0;
    for (int i = 0; i < N; i++) {
        if (modelo->alpha[i] > 1e-5) sv_count++;
    }
    printf("\nVetores de Suporte Sustentando a Margem: %d de %d dados.\n", sv_count, N);
    
    liberar_svm(modelo);
    free(X);
    free(y);
    
    printf("=========================================\n");
    return 0;
}

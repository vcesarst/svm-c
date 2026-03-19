#include "funcoes.h"

// --- FUNCOES DE I/O E GERENCIAMENTO DE MEMORIA ---

int ler_dataset(const char *nome_arquivo, double **X, double **y, int *N, int d) {
    FILE *fp = fopen(nome_arquivo, "r");
    if (!fp) {
        printf("Erro: Arquivo %s nao encontrado.\n", nome_arquivo);
        return 0;
    }

    char buffer[1024];
    int linhas = 0;
    
    // Ignora o cabecalho
    if (fgets(buffer, sizeof(buffer), fp) == NULL) {
        fclose(fp);
        return 0;
    }

    // Conta amostras para alocar memoria dinamicamente
    while (fgets(buffer, sizeof(buffer), fp)) {
        if (buffer[0] != '\n' && buffer[0] != '\0') linhas++;
    }

    *N = linhas;
    *X = (double*) malloc((*N) * d * sizeof(double));
    *y = (double*) malloc((*N) * sizeof(double));

    // Retorna ao inicio e pula o cabecalho novamente
    rewind(fp);
    fgets(buffer, sizeof(buffer), fp);

    // Leitura dos dados usando espacos como delimitadores
    for (int i = 0; i < *N; i++) {
        for (int j = 0; j < d; j++) {
            fscanf(fp, "%lf", &((*X)[i * d + j]));
        }
        fscanf(fp, "%lf", &((*y)[i]));
    }

    fclose(fp);
    return 1;
}

SVM_Model* alocar_svm(int N, int d, double C, double tol, int max_passes) {
    SVM_Model *modelo = (SVM_Model*) malloc(sizeof(SVM_Model));
    modelo->N = N;
    modelo->d = d;
    modelo->C = C;
    modelo->tol = tol;
    modelo->max_passes = max_passes;
    modelo->b = 0.0;
    
    // calloc garante inicializacao com zeros
    modelo->alpha = (double*) calloc(N, sizeof(double));
    modelo->w = (double*) calloc(d, sizeof(double));
    return modelo;
}

void liberar_svm(SVM_Model *modelo) {
    if (modelo) {
        free(modelo->alpha);
        free(modelo->w);
        free(modelo);
    }
}

// --- FUNCOES MATEMATICAS DO SMO ---

int rand_j(int i, int max) {
    int j = i;
    while (j == i) j = rand() % max;
    return j;
}

double clip_alpha(double alpha, double L, double H) {
    if (alpha > H) return H;
    if (alpha < L) return L;
    return alpha;
}

double kernel_linear(const double *X, int i, int j, int d) {
    double dot = 0.0;
    for (int k = 0; k < d; k++) {
        dot += X[i * d + k] * X[j * d + k]; // Acesso a memoria contigua 1D
    }
    return dot;
}

double calcular_f(SVM_Model *modelo, const double *X, const double *y, int idx) {
    double f = 0.0;
    for (int i = 0; i < modelo->N; i++) {
        if (modelo->alpha[i] > 0) { // Otimizacao: so processa vetores de suporte
            f += modelo->alpha[i] * y[i] * kernel_linear(X, i, idx, modelo->d);
        }
    }
    return f + modelo->b;
}

void treinar_smo(SVM_Model *modelo, const double *X, const double *y) {
    int passes = 0;
    int N = modelo->N;
    double C = modelo->C;
    double tol = modelo->tol;
    
    while (passes < modelo->max_passes) {
        int num_changed_alphas = 0;
        
        for (int i = 0; i < N; i++) {
            double E_i = calcular_f(modelo, X, y, i) - y[i];
            
            // Analise das condicoes KKT
            if ((y[i] * E_i < -tol && modelo->alpha[i] < C) || 
                (y[i] * E_i > tol && modelo->alpha[i] > 0)) {
                
                int j = rand_j(i, N);
                double E_j = calcular_f(modelo, X, y, j) - y[j];
                
                double alpha_i_old = modelo->alpha[i];
                double alpha_j_old = modelo->alpha[j];
                
                double L, H;
                if (y[i] != y[j]) {
                    L = fmax(0.0, modelo->alpha[j] - modelo->alpha[i]);
                    H = fmin(C, C + modelo->alpha[j] - modelo->alpha[i]);
                } else {
                    L = fmax(0.0, modelo->alpha[i] + modelo->alpha[j] - C);
                    H = fmin(C, modelo->alpha[i] + modelo->alpha[j]);
                }
                
                if (L == H) continue;
                
                double eta = 2.0 * kernel_linear(X, i, j, modelo->d) - 
                             kernel_linear(X, i, i, modelo->d) - 
                             kernel_linear(X, j, j, modelo->d);
                             
                if (eta >= 0) continue;
                
                modelo->alpha[j] -= (y[j] * (E_i - E_j)) / eta;
                modelo->alpha[j] = clip_alpha(modelo->alpha[j], L, H);
                
                if (fabs(modelo->alpha[j] - alpha_j_old) < 1e-5) continue;
                
                modelo->alpha[i] += y[i] * y[j] * (alpha_j_old - modelo->alpha[j]);
                
                double b1 = modelo->b - E_i - y[i] * (modelo->alpha[i] - alpha_i_old) * kernel_linear(X, i, i, modelo->d) 
                                            - y[j] * (modelo->alpha[j] - alpha_j_old) * kernel_linear(X, i, j, modelo->d);
                double b2 = modelo->b - E_j - y[i] * (modelo->alpha[i] - alpha_i_old) * kernel_linear(X, i, j, modelo->d) 
                                            - y[j] * (modelo->alpha[j] - alpha_j_old) * kernel_linear(X, j, j, modelo->d);
                
                if (0 < modelo->alpha[i] && modelo->alpha[i] < C) modelo->b = b1;
                else if (0 < modelo->alpha[j] && modelo->alpha[j] < C) modelo->b = b2;
                else modelo->b = (b1 + b2) / 2.0;
                
                num_changed_alphas++;
            }
        }
        if (num_changed_alphas == 0) passes++;
        else passes = 0;
    }
    
    // Reconstrucao do vetor normal geometrico w
    for (int k = 0; k < modelo->d; k++) {
        modelo->w[k] = 0.0;
        for (int i = 0; i < N; i++) {
            if (modelo->alpha[i] > 0) {
                modelo->w[k] += modelo->alpha[i] * y[i] * X[i * modelo->d + k];
            }
        }
    }
}

double prever(SVM_Model *modelo, const double *x_teste) {
    double soma = 0.0;
    for (int k = 0; k < modelo->d; k++) soma += modelo->w[k] * x_teste[k];
    soma += modelo->b;
    return (soma >= 0.0) ? 1.0 : -1.0;
}

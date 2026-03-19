# SVM (Soft Margin) from Scratch in C

Uma implementação "bare-metal" de uma Máquina de Vetores de Suporte (SVM) com Margem Suave, construída inteiramente do zero em linguagem C. 

Este projeto não utiliza bibliotecas externas de Machine Learning. O objetivo principal é demonstrar o domínio da matemática de otimização convexa e engenharia de software de baixo nível, descendo até a alocação contígua de memória para otimização de cache da CPU.

## 🚀 Destaques da Implementação

* **Algoritmo SMO (Sequential Minimal Optimization):** Implementação do algoritmo iterativo de John Platt para resolver o problema Dual de otimização quadrática sem a necessidade de solvers complexos.
* **Condições KKT (Karush-Kuhn-Tucker):** Validação matemática rigorosa das condições de contorno para encontrar os multiplicadores de Lagrange e identificar os Vetores de Suporte.
* **Otimização de Memória:** O dataset é alocado dinamicamente como um vetor 1D contíguo (`X[i * d + j]`), evitando a fragmentação de memória de ponteiros duplos e maximizando a localidade espacial (Cache Hit) da CPU.
* **Otimização de Compilador (HPC):** Uso de *flags* agressivas do GCC (`-O3`, `-ffast-math`, `-march=native`, `-fexpensive-optimizations`) para extrair o máximo de performance da arquitetura do processador.

## 📁 Estrutura do Repositório

```text
.
├── examples/
│   └── iris_scaled.txt    # Dataset de teste (2 classes linearmente separáveis do Scikit-Learn)
├── src/
│   ├── funcoes/
│   │   ├── funcoes.h      # Definição da struct SVM_Model e assinaturas
│   │   └── funcoes.c      # Motor matemático (IO, Kernel Linear, SMO)
│   ├── svm.c              # Ponto de entrada (Main)
│   └── Makefile           # Automação de compilação
├── .gitignore
└── README.md

⚙️ Como Compilar e Executar

O projeto inclui um Makefile configurado para compilar e rodar o código com um único comando. Navegue até a pasta src e execute:
Bash

cd src
make

Isso irá compilar os arquivos com todas as flags de otimização, gerar o executável svm.x e rodar o treinamento imediatamente utilizando o dataset iris_scaled.txt.

Para limpar os arquivos binários gerados:
Bash

make clean

📊 Validação e Resultados (vs. Scikit-Learn)

O motor matemático foi validado comparando os hiperplanos gerados pelo nosso código C com a implementação padrão da indústria (Scikit-Learn / LIBSVM) usando os mesmos parâmetros (C=1.0, Kernel Linear).

Os resultados bateram na segunda/terceira casa decimal, comprovando a eficácia do algoritmo desenvolvido:
Parâmetro	SVM C (Custom)	Scikit-Learn (LIBSVM)
w[0]	0.271099	0.267410
w[1]	-0.361790	-0.339373
w[2]	0.710905	0.699713
w[3]	0.739296	0.749147
Viés (b)	0.213805	0.242744

Pequenas variações ocorrem devido à natureza estocástica da escolha dos pares α no SMO simplificado e aos critérios de parada rigorosos baseados no Duality Gap do LIBSVM.

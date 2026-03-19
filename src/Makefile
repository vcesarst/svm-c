# Variaveis de compilacao com flags agressivas de otimizacao
CC = gcc
CFLAGS = -O3 -ffast-math -march=native -fexpensive-optimizations
LDFLAGS = -lm

# Arquivos e alvo
TARGET = svm.x
SRCS = svm.c funcoes/funcoes.c

# O primeiro alvo eh o padrao. 'make' vai chamar o 'run'.
all: run

# Regra para compilar o executavel (so compila se os arquivos .c mudarem)
$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) $(SRCS) -o $(TARGET) $(LDFLAGS)

# Regra para executar. Ela depende do $(TARGET) estar pronto.
run: $(TARGET)
	./$(TARGET)

# Limpeza dos binarios
clean:
	rm -f $(TARGET)

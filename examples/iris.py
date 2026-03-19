#importando as bibliotecas necessárias para obter o dataset iris e para poder escalar e rotular para classificação binárias as classes setosa e versicolor
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, LabelEncoder
#carregando o dataset iris
iris = load_iris()
#obtendo as características e os rótulos do dataset
X = iris.data
y = iris.target
#filtrando apenas as classes setosa e versicolor para classificação binária
X = X[y != 2]
y = y[y != 2]
#escalando as características para melhorar o desempenho do modelo
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
#rotulando as classes setosa e versicolor como -1 e 1, respectivamente
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_encoded = y_encoded * 2 - 1  # Convertendo para -1 e 1
#imprimindo as características escaladas e os rótulos codificados
print("Características escaladas:\n", X_scaled)
print("Rótulos codificados:\n", y_encoded)

#preciso salvar as variáveis X_scaled e y_encoded para usar posteriormente no treinamento do modelo de classificação binária. Posso salvar os dados em um arquivo CSV usando a biblioteca pandas ou em um arquivo numpy usando a biblioteca numpy. Aqui está um exemplo de como salvar os dados em um arquivo CSV:
import pandas as pd
#criando um DataFrame com as características escaladas e os rótulos codificados
df = pd.DataFrame(X_scaled, columns=iris.feature_names[:4])
df['target'] = y_encoded
#salvando o DataFrame em um arquivo CSV
df.to_csv('iris_scaled.csv', index=False)

#testar um SVM na proporção de 80% para treinamento e 20% para teste
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
#dividindo os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
#criando um modelo SVM e treinando-o com os dados de treinamento
model = SVC(kernel='linear')
model.fit(X_train, y_train)
#avaliando o modelo com os dados de teste
accuracy = model.score(X_test, y_test)
print("Acurácia do modelo SVM:", accuracy)
#calcular o f1 score do modelo
from sklearn.metrics import f1_score
y_pred = model.predict(X_test)
f1 = f1_score(y_test, y_pred)
print("F1 Score do modelo SVM:", f1)

#realizar a validação cruzada para avaliar o desempenho do modelo SVM
from sklearn.model_selection import cross_val_score
#realizando a validação cruzada com 5 folds
cv_scores = cross_val_score(model, X_scaled, y_encoded, cv=5)
print("Acurácia média da validação cruzada:", cv_scores.mean())

#salvar a matriz de confusão do modelo SVM em um .png
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
#calculando a matriz de confusão
cm = confusion_matrix(y_test, y_pred)
#plotando a matriz de confusão
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Matriz de Confusão do Modelo SVM')
#salvando a figura em um arquivo .png
plt.savefig('confusion_matrix_svm.png')
plt.show()

# --- EXTRAINDO O GABARITO DA RETA PARA O C ---

# O coef_ retorna uma matriz, pegamos a primeira linha [0]
vetor_w = model.coef_[0]

# O intercept_ retorna um array, pegamos o primeiro valor [0]
bias_b = model.intercept_[0]

print("\n=== GABARITO PARA O CÓDIGO EM C ===")
print("Vetor de Pesos (w):")
for i, peso in enumerate(vetor_w):
    print(f"  w[{i}] = {peso:.6f}")

print(f"\nBias (b): \n  b = {bias_b:.6f}") # O intercept_ às vezes é array de array dependendo da versão
# Se der erro no bias_b[0], use apenas bias_b

# 1 - Importando Libs usadas para análise gráfica
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 2 - Lendo a base de dados.csv e criando um dataframe
dataframe = pd.read_csv("dados/base_dados.csv")

# 3 - Pré processamento dos dados

# Removendo dados nulos (transformando em 0)
if dataframe.isnull().sum().sum() > 0:
  dataframe.fillna(0)

# Transformando colunas categóricas em boleanas
dataframe = pd.get_dummies(dataframe, columns=['gênero', 'fumante', 'região'], drop_first=True)

# Exibindo as 10 primeiras entradas (para ver se aparentemente está tudo ok 😜)
dataframe.head(10)

# 4 - Função para gerar estatísticas gerais sobre o dataset
dataframe.describe()

# 5 - Alterando o estilo dos gráficos para facilitar visualização
sns.set(style="darkgrid")

# 6 - Análises gráficas: insights sobre os dados

# Mapa de Correlação
plt.figure(figsize=(10, 6))
sns.heatmap(dataframe.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Mapa de Correlação')
plt.show()

# Dispersão: IMC vs Encargos
plt.figure(figsize=(8, 5))
sns.scatterplot(data=dataframe, x='imc', y='encargos')
plt.title('IMC vs Encargos')
plt.show()

# Dispersão: Idade vs Encargos
plt.figure(figsize=(8, 5))
sns.scatterplot(data=dataframe, x='idade', y='encargos')
plt.title('Idade vs Encargos')
plt.show()

# Dispersão: Filhos vs Encargos
plt.figure(figsize=(8, 5))
sns.scatterplot(data=dataframe, x='filhos', y='encargos')
plt.title('Filhos vs Encargos')
plt.show()

# Boxplot: Fumante vs Encargos
plt.figure(figsize=(8, 5))
sns.boxplot(x='fumante_sim', y='encargos', data=dataframe)
plt.title('Encargos por Fumante')
plt.show()

# Boxplot: Gênero vs Encargos
plt.figure(figsize=(8, 5))
sns.boxplot(x='gênero_masculino', y='encargos', data=dataframe)
plt.title('Encargos por Gênero (0 = feminino, 1 = masculino)')
plt.show()

# Boxplot: Região Sudoeste vs Encargos
plt.figure(figsize=(8, 5))
sns.boxplot(x='região_sudoeste', y='encargos', data=dataframe)
plt.title('Encargos por Região Sudoeste')
plt.show()

# Boxplot: Região Sudeste vs Encargos
plt.figure(figsize=(8, 5))
sns.boxplot(x='região_sudeste', y='encargos', data=dataframe)
plt.title('Encargos por Região Sudeste')
plt.show()

# Boxplot: Região Noroeste vs Encargos
plt.figure(figsize=(8, 5))
sns.boxplot(x='região_noroeste', y='encargos', data=dataframe)
plt.title('Encargos por Região Noroeste')
plt.show()

# Pairplot com variáveis principais
sns.pairplot(dataframe[['idade', 'imc', 'filhos', 'encargos']])
plt.suptitle('Gráficos de Dispersão em Pares', y=1.02)
plt.show()

# 7 - Libs usadas para criar o modelo preditivo

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 8 - Treinando o modelo

# Modelagem preditiva com Regressão Linear
X = dataframe.drop('encargos', axis=1)
y = dataframe['encargos']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = LinearRegression()
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)

# Avaliação
print("MSE:", mean_squared_error(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))

# Gráfico: Preço Real vs Previsto
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Valor Real")
plt.ylabel("Valor Previsto")
plt.title("Valor Real vs Valor Previsto - Regressão Linear")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.show()

# 9 - Libs de validação estatística

import statsmodels.api as sm

# 12. Validação estatística com statsmodels

# Convertendo colunas boleanas para integers
dataframe_numeric = dataframe.astype(int, errors='ignore')
X = dataframe_numeric.drop('encargos', axis=1)
y = dataframe_numeric['encargos']

X_const = sm.add_constant(X)
modelo_ols = sm.OLS(y, X_const).fit()

print(modelo_ols.summary())
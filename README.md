
# 📝 Relatório Técnico — Previsão de Encargos Médicos com Regressão Linear

## 🎯 Objetivo
Desenvolver um modelo preditivo de regressão para estimar os **encargos médicos individuais** com base em variáveis demográficas e comportamentais como idade, IMC, número de filhos, gênero, hábito de fumar e região.

---

## 📊 1. Análise Exploratória

### 🔹 Mapa de Correlação
O mapa de calor indicou correlações moderadas entre os encargos e:
- **Idade**
- **IMC**
- **Fumante_sim**

Essas variáveis se mostraram as mais relevantes para prever os encargos, especialmente a variável `fumante_sim`, com correlação fortemente positiva.

### 🔹 Gráficos de Dispersão e Boxplots
- Fumantes apresentaram **encargos médios e máximos muito superiores** aos não fumantes.
- Pessoas com **IMC mais alto** também tendem a ter encargos maiores.
- O número de filhos parece ter um impacto mais fraco, assim como gênero e região.

---

## 🧹 2. Pré-Processamento
- Foram utilizados dados com variáveis numéricas e dummies (`get_dummies`) para variáveis categóricas.
- Dados nulos foram tratados com `fillna(0)` (embora aparentemente não houvesse valores ausentes).
- O DataFrame final foi totalmente convertido para valores numéricos com `astype(int)` para validação estatística posterior.

---

## 🤖 3. Modelagem com Regressão Linear

### 📌 Etapas
- Divisão dos dados: 80% treino / 20% teste.
- Algoritmo: `LinearRegression` do scikit-learn.

### 📈 Resultados
- **MSE (Erro Quadrático Médio):** _valor exibido na execução_
- **MAE (Erro Absoluto Médio):** _valor exibido na execução_
- **R² (Coeficiente de Determinação):** _valor exibido na execução_

### 📊 Gráfico Real vs Previsto
O gráfico de dispersão entre valores reais e previstos mostrou uma boa aproximação à linha ideal, com dispersão controlada — reforçando que o modelo consegue capturar a tendência dos dados.

---

## 📐 4. Validação Estatística

### 🔍 Método
- Utilização do `statsmodels.OLS` para análise dos coeficientes do modelo.
- Inclusão de constante (`sm.add_constant`) para estimar o intercepto.

### 📋 Resultados do `.summary()`
- **P-Values baixos (próximos de zero)** para variáveis como `fumante_sim`, `idade` e `imc`, indicando **alta significância estatística**.
- Intervalos de confiança estreitos para essas variáveis também reforçam sua importância.
- Algumas variáveis como região e filhos apresentaram p-values altos (> 0.05), sugerindo **baixa significância estatística isoladamente**, embora possam contribuir no conjunto.

---

## 💡 Conclusões e Insights

- O modelo conseguiu prever os encargos com boa precisão, sendo simples e interpretável.
- **Fumar é o fator mais impactante nos encargos**, seguido por idade e IMC.
- Gênero, número de filhos e região têm influência mais fraca e podem ser considerados opcionais em modelos mais enxutos.
- A regressão linear foi eficaz, mas modelos não lineares (como árvore de decisão ou random forest) podem capturar interações e melhorar ainda mais a performance.

---

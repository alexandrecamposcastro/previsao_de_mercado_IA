# Previsao de Precos de Acoes com LSTM e yfinance + RSI + Médias Móveis

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from datetime import date, timedelta

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Input
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#  INPUT DO USUARIO 
ativo = input("Digite o ticker do ativo (ex: AAPL, GOOG, PETR4.SA): ").upper()

# DOWNLOAD DOS DADOS ATE HOJE
hoje = date.today()
print(f"Baixando dados do ativo {ativo} até {hoje}...")
dados = yf.download(ativo, start="2012-01-01", end=str(hoje))

# Verifica se o ativo foi encontrado
if dados.empty:
    print(f"Erro: Nenhum dado encontrado para o ativo {ativo}.")
    exit()

# Verifica se 'Adj Close' existe; se não, usa 'Close'
coluna_preco = 'Adj Close' if 'Adj Close' in dados.columns else 'Close'
precos = dados[[coluna_preco]].values

# CALCULO DO RSI
def calcular_rsi(series, periodo=14):
    delta = series.diff()
    ganho = (delta.where(delta > 0, 0)).rolling(window=periodo).mean()
    perda = (-delta.where(delta < 0, 0)).rolling(window=periodo).mean()
    rs = ganho / perda
    rsi = 100 - (100 / (1 + rs))
    return rsi

dados['RSI'] = calcular_rsi(dados[coluna_preco])

# CALCULO DAS MÉDIAS MÓVEIS
dados['SMA_20'] = dados[coluna_preco].rolling(window=20).mean()
dados['EMA_20'] = dados[coluna_preco].ewm(span=20, adjust=False).mean()

# Remove valores nulos criados pelas médias móveis e RSI
dados.dropna(inplace=True)

# Atualiza os preços e features
precos = dados[[coluna_preco]].values
rsi = dados[['RSI']].values
sma_20 = dados[['SMA_20']].values
ema_20 = dados[['EMA_20']].values

# Normaliza todas as features
sc_precos = MinMaxScaler()
sc_rsi = MinMaxScaler()
sc_sma = MinMaxScaler()
sc_ema = MinMaxScaler()

precos_normalizados = sc_precos.fit_transform(precos)
rsi_normalizado = sc_rsi.fit_transform(rsi)
sma_normalizado = sc_sma.fit_transform(sma_20)
ema_normalizado = sc_ema.fit_transform(ema_20)

# Criacao das janelas com features (preco, rsi, sma, ema)
X, y = [], []
for i in range(60, len(precos_normalizados)):
    X.append(np.column_stack((
        precos_normalizados[i-60:i, 0],
        rsi_normalizado[i-60:i, 0],
        sma_normalizado[i-60:i, 0],
        ema_normalizado[i-60:i, 0]
    )))
    y.append(precos_normalizados[i, 0])

X, y = np.array(X), np.array(y)

# Dividir entre treino e teste (80% treino, 20% teste)
split = int(len(X) * 0.8)
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

#  MODELO LSTM 
modelo = Sequential([
    Input(shape=(60, 4)),
    LSTM(units=50, return_sequences=True),
    Dropout(0.2),
    LSTM(units=50, return_sequences=True),
    Dropout(0.2),
    LSTM(units=50, return_sequences=True),
    Dropout(0.2),
    LSTM(units=50),
    Dropout(0.2),
    Dense(units=1)
])

modelo.compile(optimizer='adam', loss='mean_squared_error')
es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=30)

print("Treinando modelo...")
modelo.fit(X_train, y_train, epochs=100, batch_size=32, callbacks=[es])

# AVALIACAO DO MODELO
print("\nAvaliando modelo com dados de teste...")
predicoes_test_normalizado = modelo.predict(X_test)
predicoes_test = sc_precos.inverse_transform(predicoes_test_normalizado.reshape(-1, 1))
y_test_real = sc_precos.inverse_transform(y_test.reshape(-1, 1))

rmse = np.sqrt(mean_squared_error(y_test_real, predicoes_test))
mae = mean_absolute_error(y_test_real, predicoes_test)
r2 = r2_score(y_test_real, predicoes_test)

print(f"\nMétricas de avaliação no conjunto de teste:")
print(f"RMSE: {rmse:.2f}")
print(f"MAE:  {mae:.2f}")
print(f"Acurácia (R²): {r2 * 100:.2f}%")

#  PREVISAO DO PROXIMO DIA REAL 
ultimos_60_precos = precos[-60:]
ultimos_60_rsi = rsi[-60:]
ultimos_60_sma = sma_20[-60:]
ultimos_60_ema = ema_20[-60:]

entrada = np.column_stack((
    sc_precos.transform(ultimos_60_precos),
    sc_rsi.transform(ultimos_60_rsi),
    sc_sma.transform(ultimos_60_sma),
    sc_ema.transform(ultimos_60_ema)
))
entrada = np.reshape(entrada, (1, 60, 4))

previsao_normalizada = modelo.predict(entrada)
previsao = sc_precos.inverse_transform(previsao_normalizada)

proximo_dia = hoje + timedelta(days=1)
print(f"\nPrevisao do preco de fechamento ajustado para o próximo dia ({proximo_dia}) do ativo {ativo}: {previsao[0][0]:.2f}")

#  GRAFICO COMPARATIVO + RSI + Médias Móveis 
plt.figure(figsize=(14, 10))

plt.subplot(3, 1, 1)
plt.plot(dados[coluna_preco][-60:], color='red', label="Preço real")
plt.plot(np.arange(59, 60), previsao, marker='o', color='blue', label="Previsão")
plt.title(f"Previsão do próximo preço - {ativo}")
plt.xlabel("Dias")
plt.ylabel("Preço ")
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(dados['RSI'], label="RSI (14 dias)", color='purple')
plt.axhline(70, color='r', linestyle='--', label='Sobrecompra (70)')
plt.axhline(30, color='g', linestyle='--', label='Sobrevenda (30)')
plt.title("Índice de Força Relativa (RSI)")
plt.xlabel("Data")
plt.ylabel("RSI")
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(dados[coluna_preco], label="Preço", color='gray')
plt.plot(dados['SMA_20'], label="SMA 20", color='orange')
plt.plot(dados['EMA_20'], label="EMA 20", color='green')
plt.title("Médias Móveis")
plt.xlabel("Data")
plt.ylabel("Preço")
plt.legend()

plt.tight_layout()
plt.show()

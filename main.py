"""
Trabalho 2 – Introdução à Computação no Mercado Financeiro
Disciplina: SSC0964 – Introdução à Computação no Mercado Financeiro
1º Semestre 2025 – Prof. Denis Fernando Wolf
Aluno: Augusto Lescura e Felipe Tanus

Descrição:
Este script integra duas abordagens com aprimoramentos:

1. ALLOCAÇÃO SISTEMÁTICA EM CLASSES DE ATIVOS:
   - Baixa dados históricos de preços mensais via yfinance (usando a coluna "Close" ajustada).
   - Reamostragem para dados mensais (obtendo o último preço de cada mês) e cálculo dos retornos.
   - Incorpora variáveis macroeconômicas reais usando os arquivos JSON salvos localmente para:
       • SELIC (série 11) – arquivo "selic.json"
       • IPCA  (série 433) – arquivo "ipca.json"
   - Reindexa os dados oficiais para combinar com as datas mensais e converte os valores (de porcentagem para decimal).
   - Define, para cada ativo, o target como o retorno do mês seguinte e otimiza um
     RandomForestRegressor utilizando GridSearchCV com TimeSeriesSplit.
   - Calcula os pesos de alocação com base nas previsões dos modelos.
   - Realiza um backtesting simulando a evolução do portfólio.

2. TRADING SISTEMÁTICO:
   - Baixa dados diários via yfinance para um ativo (ex.: AAPL).
   - Calcula indicadores técnicos:
       • Nuvem Ichimoku (para identificar tendências e zonas de suporte/resistência)
       • Canal de Donchian (para definir limites de preço)
       • Bandas de Bollinger (para medir volatilidade, calculadas a partir de arrays 1D)
       • Média Móvel do Volume (indicador complementar)
   - Define o target binário com base no retorno do próximo dia (sinal "1" se > 0,1%; caso contrário, "0").
   - Otimiza um RandomForestClassifier com GridSearchCV e TimeSeriesSplit para gerar sinais de trading.
   - Realiza um backtesting simulando o retorno cumulativo da estratégia de trading.
"""

import json
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, accuracy_score

# =============================================================================
# FUNÇÃO PARA CARREGAR ARQUIVOS JSON DOS DADOS DO BACEN
# =============================================================================
def load_bacen_file(filepath):
    """
    Carrega uma série histórica salva localmente em formato JSON.

    Parâmetros:
      filepath (str): Caminho para o arquivo JSON.

    Retorna:
      DataFrame: Série histórica com índice de datas (convertidas para datetime) e coluna "valor".
    """
    df = pd.read_json(filepath)
    # Se a coluna de data estiver com D maiúsculo, renomeia para minúsculo "data"
    if 'data' not in df.columns and 'Data' in df.columns:
        df.rename(columns={'Data': 'data'}, inplace=True)
    if 'data' not in df.columns:
        raise KeyError("Chave 'data' não encontrada no arquivo JSON. Verifique as colunas disponíveis.")
    df['data'] = pd.to_datetime(df['data'], dayfirst=True)
    df['valor'] = pd.to_numeric(df['valor'], errors='coerce')
    df = df.set_index('data')
    return df

# =============================================================================
# FUNÇÕES DE BACKTESTING
# =============================================================================

def backtest_allocation(weights, df_all, tickers):
    """
    Simula a evolução do portfólio mensal utilizando os pesos calculados e os retornos reais.

    Parâmetros:
      weights (DataFrame): Pesos de alocação para cada ativo indexados por data.
      df_all (DataFrame): Dados dos retornos mensais e variáveis macroeconômicas.
      tickers (list): Lista dos ativos utilizados.

    Retorna:
      DataFrame: Evolução do portfólio, assumindo capital inicial igual a 1.
    """
    portfolio_returns = (df_all[tickers] * weights).sum(axis=1)
    portfolio_value = (1 + portfolio_returns).cumprod()
    return pd.DataFrame({'portfolio_value': portfolio_value})


def backtest_trading(data_daily, transaction_cost=0.001):
    """
    Simula a performance da estratégia de trading diária, considerando custos de transação.

    Parâmetros:
      data_daily (DataFrame): Deve conter as colunas "retorno" e "signal" (sinal de trading).
      transaction_cost (float): Custo percentual por transação (ex.: 0.001 equivale a 0,1%).

    Retorna:
      DataFrame: Retorno cumulativo da estratégia de trading.
    """
    data_daily['trade_entry'] = (data_daily['signal'].diff() == 1).astype(int)
    data_daily['strategy_return'] = data_daily['retorno'] * data_daily['signal'] - data_daily['trade_entry'] * transaction_cost
    data_daily['cumulative_return'] = (1 + data_daily['strategy_return']).cumprod()
    return data_daily[['cumulative_return']]


# =============================================================================
# ESTRATÉGIA DE ALOCAÇÃO SISTEMÁTICA
# =============================================================================

def allocation_systematic():
    """
    Executa a estratégia de alocação sistemática.

    Procedimentos:
      1. Baixa dados históricos de preços mensais via yfinance (usando "Close" ajustado).
      2. Reamostragem para dados mensais (último preço do mês) e cálculo dos retornos.
      3. Incorpora variáveis macroeconômicas reais utilizando os arquivos JSON:
         - SELIC (série 11) – arquivo "selic_2023.json"
         - IPCA  (série 433) – arquivo "ipca_2023.json"
      4. Reindexa os dados oficiais para coincidir com as datas mensais e converte os valores (para decimal).
      5. Define o target para cada ativo como o retorno do mês seguinte.
      6. Otimiza um RandomForestRegressor para cada ativo com GridSearchCV (TimeSeriesSplit).
      7. Calcula os pesos de alocação com base nas previsões dos modelos.

    Retorna:
      tuple: (weights, predicted_returns, df_all, tickers)
    """
    print(">>> Executando Estratégia de Alocação Sistemática...\n")
    
    tickers = ['SPY', 'IEF', 'GLD', 'VNQ']
    start_date = '2010-01-01'
    end_date = '2023-12-31'
    
    # Baixa os dados históricos ajustados dos ativos
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)['Close']
    data_monthly = data.resample('ME').last()
    returns = data_monthly.pct_change().dropna()
    
    # --- Dados Macroeconômicos Oficiais Via Arquivos JSON ---
    # Define o intervalo de datas a partir dos dados mensais
    start_date_macro = data_monthly.index.min().strftime('%d/%m/%Y')
    end_date_macro = data_monthly.index.max().strftime('%d/%m/%Y')
    
    # Carrega os dados da SELIC e do IPCA a partir dos arquivos locais
    df_selic = load_bacen_file("selic_2023.json")
    df_ipca = load_bacen_file("ipca_2023.json")
    
    # Reindexa os dados oficiais para as datas mensais (forward fill)
    df_selic = df_selic.reindex(data_monthly.index, method='ffill')
    df_ipca = df_ipca.reindex(data_monthly.index, method='ffill')
    
    # Cria o DataFrame de variáveis macroeconômicas e converte os valores (dividindo por 100)
    macro = pd.DataFrame(index=data_monthly.index)
    macro['taxa_juros'] = df_selic['valor'] / 100    # Ex.: 13,75% ⇨ 0.1375
    macro['inflacao'] = df_ipca['valor'] / 100         # Ex.: 5,8%  ⇨ 0.058
    
    # Combina os retornos dos ativos com os dados macroeconômicos
    df_all = returns.copy()
    df_all['taxa_juros'] = macro['taxa_juros']
    df_all['inflacao'] = macro['inflacao']
    
    # Define o target para cada ativo como o retorno do mês seguinte
    for asset in tickers:
        df_all[f'{asset}_target'] = df_all[asset].shift(-1)
    df_all = df_all.dropna()
    
    allocation_predictions = {}
    
    # Para cada ativo, otimiza um modelo de regressão
    for asset in tickers:
        features = df_all[[asset, 'taxa_juros', 'inflacao']]
        target = df_all[f'{asset}_target']
        
        split_index = int(0.7 * len(features))
        X_train = features.iloc[:split_index]
        X_test  = features.iloc[split_index:]
        y_train = target.iloc[:split_index]
        y_test  = target.iloc[split_index:]
        
        param_grid_reg = {
            'n_estimators': [50, 100],
            'max_depth': [None, 5, 10]
        }
        tscv = TimeSeriesSplit(n_splits=5)
        grid_reg = GridSearchCV(RandomForestRegressor(random_state=42),
                                param_grid=param_grid_reg,
                                cv=tscv,
                                scoring='neg_mean_squared_error')
        grid_reg.fit(X_train, y_train)
        best_model = grid_reg.best_estimator_
        
        pred_test = best_model.predict(X_test)
        mse = mean_squared_error(y_test, pred_test)
        print(f'Ativo: {asset} | Best Params: {grid_reg.best_params_} | MSE no teste: {mse:.4f}')
        
        predictions = best_model.predict(features)
        allocation_predictions[asset] = predictions
    
    predicted_returns = pd.DataFrame(allocation_predictions, index=df_all.index)
    
    # Calcula os pesos de alocação: se as previsões forem positivas, aloca proporcionalmente. Caso contrário, usa alocação igualitária.
    def compute_weights(row):
        preds = row.copy().where(row > 0, other=0)
        total = preds.sum()
        if total == 0:
            return pd.Series([1/len(row)] * len(row), index=row.index)
        return preds / total
    
    weights = predicted_returns.apply(compute_weights, axis=1)
    
    print("\nExemplo de alocação de pesos nos períodos finais:")
    print(weights.tail())
    
    return weights, predicted_returns, df_all, tickers


# =============================================================================
# ESTRATÉGIA DE TRADING SISTEMÁTICO
# =============================================================================

def trading_systematic():
    """
    Executa a estratégia de trading sistemático:

      1. Baixa dados diários do ativo (ex.: AAPL) com yfinance.
      2. Calcula indicadores técnicos:
         - Nuvem Ichimoku: identifica tendências e zonas de suporte/resistência.
         - Canal de Donchian: determina os limites superior e inferior em um período.
         - Bandas de Bollinger: calcula a volatilidade a partir da média móvel e do desvio padrão.
         - Média Móvel do Volume: indicador complementar para análise de volume.
      3. Define o target binário: se o retorno do próximo dia for > 0,1%, sinal "1"; senão, "0".
      4. Otimiza um RandomForestClassifier com GridSearchCV (TimeSeriesSplit) para gerar sinais.
      5. Retorna o DataFrame diário com os sinais e indicadores calculados.
    """
    print("\n>>> Executando Estratégia de Trading Sistemático...\n")
    
    ticker = 'AAPL'
    data_daily = yf.download(ticker, start='2010-01-01', end='2023-12-31', auto_adjust=True)
    data_daily = data_daily[['Open', 'High', 'Low', 'Close', 'Volume']]
    
    # Indicador 1: Nuvem Ichimoku
    def ichimoku(df, tenkan_window=9, kijun_window=26, senkou_span_b_window=52):
        high_9 = df['High'].rolling(window=tenkan_window).max()
        low_9 = df['Low'].rolling(window=tenkan_window).min()
        df['tenkan_sen'] = (high_9 + low_9) / 2

        high_26 = df['High'].rolling(window=kijun_window).max()
        low_26 = df['Low'].rolling(window=kijun_window).min()
        df['kijun_sen'] = (high_26 + low_26) / 2

        df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(kijun_window)
        
        high_52 = df['High'].rolling(window=senkou_span_b_window).max()
        low_52 = df['Low'].rolling(window=senkou_span_b_window).min()
        df['senkou_span_b'] = ((high_52 + low_52) / 2).shift(kijun_window)
        return df

    data_daily = ichimoku(data_daily)
    
    # Indicador 2: Canal de Donchian
    def donchian_channel(df, window=20):
        df['donchian_upper'] = df['High'].rolling(window=window).max()
        df['donchian_lower'] = df['Low'].rolling(window=window).min()
        df['donchian_range'] = df['donchian_upper'] - df['donchian_lower']
        return df

    data_daily = donchian_channel(data_daily)
    
    # Indicador 3: Bandas de Bollinger
    def bollinger_band_width(df, window=20, num_std=2):
        close_vals = df['Close'].to_numpy().flatten()
        sma = pd.Series(close_vals).rolling(window=window).mean().to_numpy().flatten()
        std = pd.Series(close_vals).rolling(window=window).std().to_numpy().flatten()
        bollinger_upper = sma + num_std * std
        bollinger_lower = sma - num_std * std
        bb_width = (bollinger_upper - bollinger_lower) / sma
        df['bollinger_upper'] = pd.Series(bollinger_upper, index=df.index)
        df['bollinger_lower'] = pd.Series(bollinger_lower, index=df.index)
        df['bollinger_bandwidth'] = pd.Series(bb_width, index=df.index)
        return df

    data_daily = bollinger_band_width(data_daily)
    
    # Indicador Complementar: Média Móvel do Volume
    def volume_indicator(df, window=20):
        df['volume_ma'] = df['Volume'].rolling(window=window).mean()
        return df

    data_daily = volume_indicator(data_daily)
    
    # Prepara o dataset e define o target de trading
    data_daily['retorno'] = data_daily['Close'].pct_change()
    data_daily = data_daily.dropna()
    
    threshold = 0.001  # Se o retorno do próximo dia for > 0,1%, sinal "1"
    data_daily['target'] = (data_daily['retorno'].shift(-1) > threshold).astype(int)
    data_daily = data_daily.dropna()
    
    features = [
        'tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b',
        'donchian_upper', 'donchian_lower', 'donchian_range',
        'bollinger_bandwidth', 'volume_ma'
    ]
    X = data_daily[features]
    y = data_daily['target']
    
    split_index = int(0.7 * len(X))
    X_train = X.iloc[:split_index]
    X_test  = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test  = y.iloc[split_index:]
    
    param_grid_clf = {
        'n_estimators': [50, 100],
        'max_depth': [None, 5, 10]
    }
    tscv = TimeSeriesSplit(n_splits=5)
    grid_clf = GridSearchCV(RandomForestClassifier(random_state=42),
                            param_grid=param_grid_clf,
                            cv=tscv,
                            scoring='accuracy')
    grid_clf.fit(X_train, y_train)
    best_clf = grid_clf.best_estimator_
    
    y_pred = best_clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Precisão do modelo de trading: {accuracy:.2f} | Best Params: {grid_clf.best_params_}')
    
    data_daily.loc[X_test.index, 'signal'] = y_pred
    
    print("\nExemplo de sinais de trading (últimas linhas):")
    print(data_daily[['Close', 'target', 'signal']].tail())
    
    return data_daily


# =============================================================================
# EXECUÇÃO DAS ESTRATÉGIAS INTEGRADAS
# =============================================================================

def main():
    weights, predicted_returns, df_all, tickers = allocation_systematic()
    
    backtest_alloc = backtest_allocation(weights, df_all, tickers)
    print("\nBacktesting da Estratégia de Alocação (Evolução do Portfólio):")
    print(backtest_alloc.tail())
    
    trading_data = trading_systematic()
    
    backtest_trade = backtest_trading(trading_data)
    print("\nBacktesting da Estratégia de Trading (Retorno Cumulativo):")
    print(backtest_trade.tail())
    
if __name__ == '__main__':
    main()

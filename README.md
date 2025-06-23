# Estratégias Sistemáticas de Alocação e Trading no Mercado Financeiro

Este projeto apresenta a implementação de duas estratégias computacionais aplicadas ao mercado financeiro:

1. **Estratégia de Alocação Sistemática em Classes de Ativos**  
   - Utiliza dados históricos mensais de preços (obtidos via yfinance) e dados macroeconômicos oficiais (taxa SELIC e IPCA) para prever os retornos futuros dos ativos.
   - Aplica modelos de Random Forest Regressor, com otimização de hiperparâmetros utilizando GridSearchCV e validação temporal via TimeSeriesSplit.
   - Calcula e exibe os pesos de alocação dinamicamente com base nas previsões.

2. **Estratégia de Trading Sistemático**  
   - Utiliza dados diários de um ativo (por exemplo, AAPL) para calcular indicadores técnicos (Ichimoku, Canal de Donchian, Bandas de Bollinger e Média Móvel do Volume).
   - Treina um Random Forest Classifier para gerar sinais de compra (target binário) e executa o backtesting da estratégia.

## Visão Geral

O projeto integra técnicas de machine learning com dados financeiros reais para auxiliar na tomada de decisões de investimentos. Os dados macroeconômicos são obtidos através de arquivos JSON previamente salvos (por exemplo, `selic.json` e `ipca.json`), os quais contêm as séries oficiais da taxa SELIC e do IPCA, respectivamente.

## Funcionalidades

- **Download de Dados Históricos:**  
  Utiliza a biblioteca *yfinance* para baixar dados de preços ajustados dos ativos para os períodos definidos.

- **Integração de Dados Macroeconômicos:**  
  Dados oficiais (SELIC e IPCA) são carregados a partir de arquivos JSON por meio da função `load_bacen_file()` e são reindexados para coincidir com as datas dos dados mensais.

- **Modelagem com Random Forest:**  
  - **Alocação:** Modelo de regressão (RandomForestRegressor) para prever os retornos futuros dos ativos e determinar a alocação do portfólio.
  - **Trading:** Modelo de classificação (RandomForestClassifier) para gerar sinais de compra com base em indicadores técnicos calculados a partir dos dados diários.

- **Cálculo de Indicadores Técnicos:**  
  Inclui o cálculo da Nuvem Ichimoku, Canal de Donchian, Bandas de Bollinger e Média Móvel do Volume.

- **Backtesting:**  
  As funções de backtesting simulam a evolução do portfólio para a estratégia de alocação e o retorno cumulativo para a estratégia de trading, considerando também os custos de transação.

## Requisitos

- Python 3.7 ou superior
- Bibliotecas:
  - `pandas`
  - `numpy`
  - `yfinance`
  - `scikit-learn`
  - `requests`
  - `bcdata` (caso deseje utilizar a abordagem via API; neste projeto os dados oficiais foram carregados via arquivos JSON)

## Instalação

1. Clone o repositório:

   git clone https://github.com/seu_usuario/projeto_financeiro.git
   cd projeto_financeiro

2. Instale as dependências:

   pip install -r requirements.txt

3. Certifique-se de que os arquivos selic.json e ipca.json estejam na raiz do projeto (ou ajuste os caminhos conforme necessário).

## Uso
Para executar o projeto, basta rodar o script principal:

python main.py

- O script executará ambas as estratégias, exibindo:

- Mensagens referentes à execução das estratégias.

- Modelos selecionados e MSEs para cada ativo na estratégia de alocação.

- Tabelas de alocação de pesos e evolução do portfólio (backtesting da alocação).

- Resultados da estratégia de trading, incluindo a acurácia do modelo, sinais gerados e o retorno cumulativo.

## Contribuição
Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou enviar pull requests se tiver sugestões de melhoria, encontrar algum bug ou quiser ampliar as funcionalidades do projeto.

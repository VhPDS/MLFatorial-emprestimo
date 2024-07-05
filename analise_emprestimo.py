# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 20:00:32 2024

@author: torug
"""

#%% Instalando os pacotes necessários

!pip install pandas
!pip install numpy
!pip install factor_analyzer
!pip install sympy
!pip install scipy
!pip install matplotlib
!pip install seaborn
!pip install plotly
!pip install pingouin
!pip install pyshp

#%% Importando os pacotes necessários

import pandas as pd
import numpy as np
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.io as pio
pio.renderers.default = 'browser'
import plotly.graph_objects as go
import sympy as sy
import scipy as sp

#%% Carregando e preparando os dados

# Carregar o dataset de empréstimos bancários
emprestimo = pd.read_excel("emprestimo_banco.xlsx")
# Fonte: adaptado de https://www.kaggle.com/datasets/itsmesunil/bank-loan-modelling

# Remover a coluna 'ID' que não é necessária para a análise
emprestimo = emprestimo.drop(columns=['ID'])

# Exibir informações gerais sobre o dataset
emprestimo.info()

# Descrever estatisticamente o dataset
emprestimo_describe = emprestimo.describe()

# Calcular a matriz de correlação entre as variáveis
corr = emprestimo.corr()

#%% Gráfico interativo de mapa de calor das correlações

# Criar uma figura interativa com Plotly
fig = go.Figure()

# Adicionar um mapa de calor das correlações
fig.add_trace(
    go.Heatmap(
        x = corr.columns,
        y = corr.index,
        z = np.array(corr),
        text=corr.values,
        texttemplate='%{text:.3f}',
        colorscale='viridis'))

# Ajustar o layout da figura
fig.update_layout(
    height = 600,
    width = 600,
    yaxis=dict(autorange="reversed"))

# Exibir a figura
fig.show()

#%% Teste de esfericidade de Bartlett

# Calcular o teste de esfericidade de Bartlett
bartlett, p_value = calculate_bartlett_sphericity(emprestimo)

# Exibir os resultados do teste
print(f'Qui² Bartlett: {round(bartlett, 2)}')
print(f'p-valor: {round(p_value, 4)}')

#%% Análise Fatorial

# Ajustar o modelo de análise fatorial com 6 fatores (sem rotação)
fa = FactorAnalyzer(n_factors=6, method='principal',rotation=None).fit(emprestimo)

# Obter os autovalores (eigenvalues)
autovalores = fa.get_eigenvalues()[0]

# Vamos adotar 2 fatores, já que somente 2 autovalores são maiores que 1
fa = FactorAnalyzer(n_factors=2, method='principal',rotation=None).fit(emprestimo)

# Obter a variância explicada pelos fatores
autovalores_fatores = fa.get_factor_variance()

# Criar uma tabela com os autovalores e a variância explicada
tabela_eigen = pd.DataFrame(autovalores_fatores)
tabela_eigen.columns = [f"Fator {i+1}" for i, v in enumerate(tabela_eigen.columns)]
tabela_eigen.index = ['Autovalor','Variância', 'Variância Acumulada']
tabela_eigen = tabela_eigen.T

# Exibir a tabela de variância explicada
print(tabela_eigen)

# Obter as cargas fatoriais
cargas_fatoriais = fa.loadings_

# Criar uma tabela com as cargas fatoriais
tabela_cargas = pd.DataFrame(cargas_fatoriais)
tabela_cargas.columns = [f"Fator {i+1}" for i, v in enumerate(tabela_cargas.columns)]
tabela_cargas.index = emprestimo.columns

# Exibir a tabela de cargas fatoriais
print(tabela_cargas)

# Obter as comunalidades
comunalidades = fa.get_communalities()

# Criar uma tabela com as comunalidades
tabela_comunalidades = pd.DataFrame(comunalidades)
tabela_comunalidades.columns = ['Comunalidades']
tabela_comunalidades.index = emprestimo.columns

# Exibir a tabela de comunalidades
print(tabela_comunalidades)

# Transformar os dados originais nos escores fatoriais
fatores = pd.DataFrame(fa.transform(emprestimo))
fatores.columns = [f"Fator {i+1}" for i, v in enumerate(fatores.columns)]

# Obter os pesos dos escores fatoriais
scores = fa.weights_

# Criar uma tabela com os pesos dos escores fatoriais
tabela_scores = pd.DataFrame(scores)
tabela_scores.columns = [f"Fator {i+1}" for i, v in enumerate(tabela_scores.columns)]
tabela_scores.index = emprestimo.columns

# Exibir a tabela de pesos dos escores fatoriais
print(tabela_scores)

#%% Rotação dos fatores utilizando varimax

# Ajustar o modelo de análise fatorial com rotação varimax
fa = FactorAnalyzer(n_factors=2, method='principal',rotation='varimax').fit(emprestimo)

# Obter a variância explicada pelos fatores rotacionados
autovalores_varimax_fatores = fa.get_factor_variance()

# Criar uma tabela com os autovalores e a variância explicada após rotação
tabela_eigen = pd.DataFrame(autovalores_fatores)
tabela_eigen.columns = [f"Fator {i+1}" for i, v in enumerate(tabela_eigen.columns)]
tabela_eigen.index = ['Autovalor','Variância', 'Variância Acumulada']
tabela_eigen = tabela_eigen.T

# Exibir a tabela de variância explicada após rotação
print(tabela_eigen)

# Obter as cargas fatoriais rotacionadas
cargas = fa.loadings_

# Criar uma tabela com as cargas fatoriais rotacionadas
tabela_cargas = pd.DataFrame(cargas)
tabela_cargas.columns = [f"Fator {i+1}" for i, v in enumerate(tabela_cargas.columns)]
tabela_cargas.index = emprestimo.columns

# Exibir a tabela de cargas fatoriais rotacionadas
print(tabela_cargas)

#%% Determinando as comunalidades após rotação

# Obter as comunalidades após rotação
comunalidades = fa.get_communalities()

# Criar uma tabela com as comunalidades após rotação
tabela_comunalidades = pd.DataFrame(comunalidades)
tabela_comunalidades.columns = ['Comunalidades']
tabela_comunalidades.index = emprestimo.columns

# Exibir a tabela de comunalidades após rotação
print(tabela_comunalidades)

#%% Adicionando os escores fatoriais ao dataset original

# Transformar os dados originais nos escores fatoriais rotacionados
fatores = pd.DataFrame(fa.transform(emprestimo))
fatores.columns =  [f"Fator {i+1}" for i, v in enumerate(fatores.columns)]

# Adicionar os escores fatoriais ao dataset original
emprestimo = pd.concat([emprestimo.reset_index(drop=True), fatores], axis=1)

# Exibir as primeiras linhas do dataset com os escores fatoriais adicionados
print(emprestimo.head())

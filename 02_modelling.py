# Databricks notebook source
! pip install mlflow
! pip install lightgbm
!pip install --upgrade pip
!pip install shap
!pip install matplotlib
!pip install seaborn

import numpy as np
import mlflow
import mlflow.lightgbm
import lightgbm as lgb
# mport shap -----erro ao importar a library shap
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# COMMAND ----------

df = pd.read_parquet("/Workspace/Repos/fraud/master-cloud/treino_processado.parquet")


# COMMAND ----------

X = df.loc[(df["type"] == 'TRANSFER') | (df["type"] == 'CASH_OUT')]

randomState = 5
np.random.seed(randomState)

Y = df["isFraud"]
del X['isFraud']

# COMMAND ----------

X

# COMMAND ----------

Y

# COMMAND ----------

X = X.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis = 1)


# COMMAND ----------

X.loc[X.type == 'TRANSFER', 'type'] = 0
X.loc[X.type == 'CASH_OUT', 'type'] = 1
X.type = X.type.astype(int) # convert dtype('O') to dtype(int)

# COMMAND ----------

Xfraud = X.loc[Y == 1]
XnonFraud = X.loc[Y == 0]
print('\nThe fraction of fraudulent transactions with \'oldBalanceDest\' = \
\'newBalanceDest\' = 0 although the transacted \'amount\' is non-zero is: {}'.\
format(len(Xfraud.loc[(Xfraud.oldbalanceDest == 0) & \
(Xfraud.newbalanceDest == 0) & (Xfraud.amount)]) / (1.0 * len(Xfraud))))

print('\nThe fraction of genuine transactions with \'oldBalanceDest\' = \
newBalanceDest\' = 0 although the transacted \'amount\' is non-zero is: {}'.\
format(len(XnonFraud.loc[(XnonFraud.oldbalanceDest == 0) & \
(XnonFraud.newbalanceDest == 0) & (XnonFraud.amount)]) / (1.0 * len(XnonFraud))))

# COMMAND ----------

X['errorBalanceOrig'] = X.newbalanceOrig + X.amount - X.oldbalanceOrg
X['errorBalanceDest'] = X.oldbalanceDest + X.amount - X.newbalanceDest

# COMMAND ----------

limit = len(X)

def plotStrip(x, y, hue, figsize = (14, 9)):
    
    fig = plt.figure(figsize = figsize)
    colours = plt.cm.tab10(np.linspace(0, 1, 9))
    with sns.axes_style('ticks'):
        ax = sns.stripplot(x, y, \
             hue = hue, jitter = 0.4, marker = '.', \
             size = 4, palette = colours)
        ax.set_xlabel('')
        ax.set_xticklabels(['genuine', 'fraudulent'], size = 16)
        for axis in ['top','bottom','left','right']:
            ax.spines[axis].set_linewidth(2)

        handles, labels = ax.get_legend_handles_labels()
        plt.legend(handles, ['Transfer', 'Cash out'], bbox_to_anchor=(1, 1), \
               loc=2, borderaxespad=0, fontsize = 16);
    return ax

# COMMAND ----------

ax = plotStrip(Y[:limit], X.step[:limit], X.type[:limit])
ax.set_ylabel('time [hour]', size = 16)
ax.set_title('Striped vs. homogenous fingerprints of genuine and fraudulent \
transactions over time', size = 20);

# COMMAND ----------

limit = len(X)
ax = plotStrip(Y[:limit], X.amount[:limit], X.type[:limit], figsize = (14, 9))
ax.set_ylabel('amount', size = 16)
ax.set_title('Same-signed fingerprints of genuine \
and fraudulent transactions over amount', size = 18);

# COMMAND ----------

# Criar o gráfico KDE
sns.kdeplot(
    data= df,
    x="type",  # Substitua "variavel1" pelo nome da primeira variável
    y="amount",  # Substitua "variavel2" pelo nome da segunda variável
    hue="isFraud",  # Colorir por categoria de output
    fill=True,
    alpha=0.5,  # Ajustar a transparência
    #palette=palette,  # Usar a paleta de cores definida (opcional)
    thresh=0.1  # Ajustar o limiar de contorno (opcional)
)

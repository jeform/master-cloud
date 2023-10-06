# Databricks notebook source
# Databricks notebook source
from pyspark.sql import functions as F
import pandas as pd

# COMMAND ----------

df = spark.read.csv("file:/Workspace/Repos/fraud/master-cloud/PS_20174392719_1491204439457_log - Copia.csv", header=True)

# COMMAND ----------

df = df.filter(F.col('isFraud') == 1)

# COMMAND ----------

display(df)

# COMMAND ----------

df = df.withColumn('amount', F.col('amount').cast('float'))
df = df.withColumn('oldbalanceOrg', F.col('oldbalanceOrg').cast('float'))
df = df.withColumn('newbalanceOrig', F.col('newbalanceOrig').cast('float'))
df = df.withColumn('oldbalanceDest', F.col('oldbalanceDest').cast('float'))
df = df.withColumn('newbalanceDest', F.col('newbalanceDest').cast('float'))
#df = df.withColumn('isFraud', F.col('isFraud').cast('int'))
df = df.withColumn('isFlaggedFraud', F.col('isFlaggedFraud').cast('int'))

# COMMAND ----------

pd = df.toPandas()

# COMMAND ----------

display(df)

# COMMAND ----------

display(df.filter(F.col('isFlaggedFraud') == 1))

# COMMAND ----------

df = df.drop(F.col('isFlaggedFraud'))
display(df)

# COMMAND ----------

df.groupby("type").agg(F.avg("amount")).display()

# COMMAND ----------

pd.describe()

# COMMAND ----------

pd['amount'].hist(log=True)

# COMMAND ----------

pd.corr(method="spearman")

# COMMAND ----------

display(df)

# COMMAND ----------

import seaborn as sns

sns.histplot(data = pd, x = '', hue='isFraud',log_scale = (True, False),alpha=0.5)

# COMMAND ----------

display(
    df
    .groupby("type","isFraud")
    .agg(F.mean('amount'))
)

# COMMAND ----------

pd.to_parquet("/Workspace/Repos/fraud/master-cloud/treino_processado.parquet")

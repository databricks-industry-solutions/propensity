# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC The following are configuration settings that will be used throughout all steps of this solution accelerator:

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
import pyspark.sql.functions as fn

# COMMAND ----------

# DBTITLE 1,Initialize Config Settings
if 'config' not in locals():
  config = {}

# COMMAND ----------

# DBTITLE 1,Accelerator path 
# Here we use a dbfs /tmp path to replace the mount point in order to reduce dependencies. Modify this to point to your own DBFS path or mount point
config['mount_point'] = '/tmp/propensity_workflow'

# COMMAND ----------

# DBTITLE 1,Database
config['database_name'] = 'propensity_workflow'

# create database 
_ = spark.sql(f"create database if not exists {config['database_name']}")

# set current datebase context
_ = spark.catalog.setCurrentDatabase(config['database_name'])

# COMMAND ----------

# DBTITLE 1,Current Day
# try get current day from dataset - this is wrapped in try-except because it fails before the transactions_adj table is created

try:
  current_day = (
      spark
        .table('transactions_adj')
        .groupBy()
          .agg(
            fn.max('day').alias('current_day')
            )
          ).collect()[0]['current_day']

  config['current_day'] = current_day
except:
  pass

# COMMAND ----------

def teardown():
  from databricks.feature_store import FeatureStoreClient
  fs = FeatureStoreClient()
  database_name = config['database_name']

  try:
    fs.drop_table(
        name=f'{database_name}.household_features'
      )
    fs.drop_table(
        name=f'{database_name}.commodity_features'
      )
    fs.drop_table(
        name=f'{database_name}.household_commodity_features'
      )
  except ValueError: 
    pass

  _ = spark.sql(f"drop database if exists {database_name} cascade")
  

# COMMAND ----------

config

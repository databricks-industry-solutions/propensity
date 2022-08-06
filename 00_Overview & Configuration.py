# Databricks notebook source
# MAGIC %md The purpose of this notebook is to capture configuration settings affecting the other notebooks in this solution accelerator. 

# COMMAND ----------

# MAGIC %md ##Introduction
# MAGIC 
# MAGIC The purpose of this solution accelerator is to demonstrate how scores estimating the probability a given household will purchase from a product category (or similar grouping of products) within a future period may be derived.  These propensity scores are commonly used in marketing workflows to determine which offers, advertisements, *etc.* to put in front of a given customer and to identify subsets of customers to target for various promotional engagements. 
# MAGIC 
# MAGIC A typical pattern in calculating propensity scores is to derive a set of features from a combination of demographic and behavioral information and then train a model to predict future purchases (or other desired responses) from these.  Scores may be calculated in real-time but quite often these are calculated in advance of the future period, persisted and retrieved for use throughout that period.
# MAGIC 
# MAGIC We may think of each of these three major activities, *i.e.* feature engineering, model training, and customer scoring, can be thought of as three distinct but related workflows to be implemented to build a sustainable propensity scoring engine.  In this solution accelerator, each is tackled in a separate notebook to help clarify the boundaries between each:</p>
# MAGIC 
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/propensity_workflow3.png' width='700'>
# MAGIC 
# MAGIC Across these notebooks, we will strive to produce propensity scores that provide us the likliehood a customer (household) will purchase products within a category in the next 30 days based on features generated from customer interactions taking place in various periods from over the last couple years.  In real-world implementations, the forward looking period may be shorter or longer depending on the specific needs driving the scoring and feature generation may be more or less exhaustive than what is shown here. Still, organizations seeking to build a robust propensity scoring pipeline should find value in the concepts explored in each stage of the demonstrated process.

# COMMAND ----------

# MAGIC %md ##Step 1: Initialize Configuration Settings
# MAGIC 
# MAGIC In this step, we simply initialize the dictionary that will contain our configuration values:

# COMMAND ----------

# DBTITLE 1,Initialize Config Settings
if 'config' not in locals():
  config = {}

# COMMAND ----------

# MAGIC %md We are going to extract user information from the notebook environment for use in later steps. You can also include user name in database and temporary folder names: 

# COMMAND ----------

# DBTITLE 1,Extract user information from the notebook environment
useremail = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
username_sql_compatible = useremail.split('@')[0].replace(".", "_")

# COMMAND ----------

# MAGIC %md ##Step 2: Database Configuration Settings
# MAGIC 
# MAGIC In this step, we define any configuration settings that control where and how table objects are created:

# COMMAND ----------

# DBTITLE 1,Database Configs
config['database'] = f'propensity'

# COMMAND ----------

# MAGIC %md ##Step 3: Storage Configuration Settings
# MAGIC 
# MAGIC In this step, we define configuration settings for accessing storage, including the name of the mount point through which we will access our data files:

# COMMAND ----------

# DBTITLE 1,External Storage Configs
config['dbfs_mount'] = '/tmp/propensity'

config['automl_data_dir'] = 'dbfs:'+ config['dbfs_mount'] + '/automl'
dbutils.fs.mkdirs(config['automl_data_dir'])

# COMMAND ----------

# MAGIC %md ##Step 4: mlflow Experiment Settings
# MAGIC 
# MAGIC In this step, we define configuration settings for an mlflow experiment:

# COMMAND ----------

import mlflow

experiment_name = f"/Users/{useremail}/propensity"
mlflow.set_experiment(experiment_name) 

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the [Databricks License](https://databricks.com/db-license-source).  All included or referenced third party libraries are subject to the licenses set forth below.

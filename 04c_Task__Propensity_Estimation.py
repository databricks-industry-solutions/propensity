# Databricks notebook source
# MAGIC %md The purpose of this notebook is to calculate propensity profiles as part of our propensity scoring work. This notebook was developed using the **Databricks 12.2 LTS ML** runtime.

# COMMAND ----------

# MAGIC %md ##Introduction
# MAGIC
# MAGIC In this notebook, we will leverage the production instances of our models to score households for commodity propensity. This step is typically run on a daily basis, following the calculation of new features for the *current day*. Scores for each commodity are recorded as a field named for that commodity in our *household_commodity_propensities* table.  Each record in that table represents a household and its associated scores.

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
import mlflow
from databricks.feature_store import FeatureStoreClient

import pyspark.sql.functions as fn

from datetime import datetime

from delta.tables import *

# COMMAND ----------

# MAGIC %md ##Step 1: Retrieve Configuration Settings
# MAGIC
# MAGIC This notebook will be typically run as part of a regularly scheduled workflow.  However, during development and initialization of the feature store, we should expect to see it run manually.  To support either scenario, we will define [widgets](https://docs.databricks.com/notebooks/widgets.html) through which values can be assigned to the notebook, either directly or through a runtime call from another notebook.  We will first attempt to retrieve configuration values from the jobs workflow but failing that, we will fallback to values supplied by these widgets:

# COMMAND ----------

# DBTITLE 1,Set Widgets (Used in Manual Runs)
dbutils.widgets.text('current day','2019-12-12') # Dec 12, 2019 is last date in our dataset
dbutils.widgets.text('database name','propensity_workflow') # Use the database name in `util/config`

# COMMAND ----------

# DBTITLE 1,Get Database Config
try:
  database_name = dbutils.jobs.taskValues.get(
    taskKey = '_Intro_and_Config',
    key = 'database_name',
    debugValue = dbutils.widgets.get('database name')
    )
except:
  database_name = dbutils.widgets.get('database name')

# set current database
_ = spark.catalog.setCurrentDatabase(database_name)

# COMMAND ----------

# DBTITLE 1,Get Current Date Config
try:
  current_day = dbutils.jobs.taskValues.get(
    taskKey = '_Intro_and_Config', 
    key = 'current_day', 
    debugValue = dbutils.widgets.get('current day')
    )
except:
  current_day = dbutils.widgets.get('current day')

# set current date to actual date value
current_day = datetime.strptime(current_day, '%Y-%m-%d').date()

# COMMAND ----------

# MAGIC %md ##Step 2: Assemble Households & Commodities to Score
# MAGIC
# MAGIC The households we wish to score are identified as follows:

# COMMAND ----------

# DBTITLE 1,Get Households to Score
households = (
  spark
    .table('transactions_adj')
    .select('household_key')
    .distinct()
  )

display(households)

# COMMAND ----------

# MAGIC %md The commodities for which we wish to score each household are as follows:
# MAGIC
# MAGIC **NOTE** The *model_uri* field will be used to help us retrieve the appropriate model as we score our data.

# COMMAND ----------

# DBTITLE 1,Get Commodities to Score
commodities = (
  spark
    .table('commodities_to_score')
    .withColumn('model_uri', fn.expr("concat( 'models:/propensity ', commodity_clean, '/Production')"))
    .select('commodity_desc','commodity_clean', 'model_uri')
  )

display(commodities)

# COMMAND ----------

# MAGIC %md We can combine these into a set of values that will determine the items we need to score:

# COMMAND ----------

# DBTITLE 1,Assemble Set of Keys to Score
items_to_score = (
  households
    .crossJoin(commodities.drop('model_uri'))
    .withColumn('day', fn.lit(current_day)) # day for linking to features
  ).cache()

display(items_to_score)

# COMMAND ----------

# MAGIC %md ##Step 3: Calculate Propensities
# MAGIC
# MAGIC To get started with the calculation of propensities, we will first setup a temporary table to which we will record the propensity scores we will generate. There are two common patterns for such a table, one within which we have a single record per household and each score is assigned to a model-specific column and another where each household and model score is its own record.  The former is a bit easier for some organizations to initially conceive, especially if we are thinking of the problem from the perspective of creating a household *profile* (as its frequently termed) while the other is technically easier to employ.  We'll create both with this exercise, referring to the first as the *pivotted* and the latter as the *unpivotted* tables:

# COMMAND ----------

# DBTITLE 1,Setup Temp Tables for Propensity Scores
_ = (
  households
    .withColumn('day', fn.lit(current_day))
    .write
      .format('delta')
      .mode('overwrite')
      .option('overwriteSchema','true')
      .saveAsTable('TEMP__household_commodity_propensities__PIVOTED')
  )

_ = (
  items_to_score
    .select(
      'household_key',
      'day',
      'commodity_desc'
    )
    .filter("1=2") # make sure its an emtpy schema at this time
    .write
      .format('delta')
      .mode('overwrite')
      .option('overwriteSchema','true')
      .saveAsTable('TEMP__household_commodity_propensities__UNPIVOTED')
  )

# COMMAND ----------

# MAGIC %md As we calculate new propensity scores, we will record these to our pivoted table using a new field.  To simplify the process of adding a field to the table's schema, we will turn on [schema autoMerge](https://docs.databricks.com/delta/update-schema.html) as follows:

# COMMAND ----------

# DBTITLE 1,Enable Schema AutoMerge
spark.conf.set("spark.databricks.delta.schema.autoMerge.enabled","true")

# COMMAND ----------

# MAGIC %md We can now perform inference on our data and append the scores as fields to our temp tables as follows:

# COMMAND ----------

# DBTITLE 1,Append Scores to Temp Tables
# connect to feature store
fs = FeatureStoreClient()

# for each commodity
for commodity in commodities.collect():

  # get details for each commodity
  commodity_desc = commodity['commodity_desc']
  commodity_clean = commodity['commodity_clean']
  model_uri = commodity['model_uri']
  print(commodity_desc)

  # identify households to score for this commodity
  batch = items_to_score.filter(f"commodity_desc='{commodity_desc}'")

  # get scores, explicitly casting data type
  scores = (
    fs
      .score_batch(model_uri, batch)
      .select('household_key','day', 'commodity_desc', 'prediction')
      .withColumn('prediction', fn.expr('CAST(1 - prediction AS DOUBLE)')) 
    )

  # add scores to pivotted table
  target = DeltaTable.forName(spark, f'{database_name}.TEMP__household_commodity_propensities__PIVOTED')
  _ = (
    target.alias('t')
      .merge(
        (
          scores
            .withColumnRenamed('prediction', commodity_clean) # rename prediction column for commodity being scored
            .drop('commodity_desc')
          ).alias('s'), 
        "t.household_key=s.household_key"
        )
      .whenMatchedUpdateAll()
      .whenNotMatchedInsertAll()
    ).execute()

  # add scores to unpivotted table
  _ = (
    scores
      .select(
          'household_key',
          'day',
          'commodity_desc',
          'prediction'
        )
      .write
      .format('delta')
      .mode('append') # equivalent to insert
      .saveAsTable('TEMP__household_commodity_propensities__UNPIVOTED')
  )


# COMMAND ----------

# MAGIC %md Our temp tables are now populated with propensity scores for each commodity. Let's review these data and complete data processing starting with our pivoted table:

# COMMAND ----------

# DBTITLE 1,Review Pivoted Propensity Scores
# MAGIC %sql
# MAGIC
# MAGIC SELECT *
# MAGIC FROM TEMP__household_commodity_propensities__PIVOTED
# MAGIC ORDER BY household_key;

# COMMAND ----------

# MAGIC %md We can now swap this table for our production-ready table with a simple SQL statement. To avoid messing up any permissions or shares on the original table that could occur if we dropped the original table and renamed the other (or otherwise issued a CREATE OR REPLACE TABLE statement), we'll move data to the production table using an INSERT OVERWRITE statement which will truncate the target table before inserting the new records.  Because we have schema evolution enabled, any new columns will be appended to the table automatically:

# COMMAND ----------

# DBTITLE 1,Elevate Temp Table Data to Production
# MAGIC %sql
# MAGIC
# MAGIC -- create empty table if needed
# MAGIC CREATE TABLE IF NOT EXISTS household_commodity_propensities__PIVOTED
# MAGIC AS
# MAGIC   SELECT * FROM TEMP__household_commodity_propensities__PIVOTED WHERE 1=2;
# MAGIC
# MAGIC -- truncate table and insert new records
# MAGIC INSERT OVERWRITE household_commodity_propensities__PIVOTED
# MAGIC SELECT * FROM TEMP__household_commodity_propensities__PIVOTED;
# MAGIC
# MAGIC -- display results
# MAGIC SELECT *
# MAGIC FROM household_commodity_propensities__PIVOTED;

# COMMAND ----------

# MAGIC %md We can now do the same for our unpivoted table:

# COMMAND ----------

# DBTITLE 1,Review Unpivoted Propensity Scores
# MAGIC %sql
# MAGIC
# MAGIC SELECT *
# MAGIC FROM TEMP__household_commodity_propensities__UNPIVOTED
# MAGIC ORDER BY household_key;

# COMMAND ----------

# DBTITLE 1,Elevate Temp Table Data to Production
# MAGIC %sql
# MAGIC
# MAGIC -- create empty table if needed
# MAGIC CREATE TABLE IF NOT EXISTS household_commodity_propensities__UNPIVOTED (
# MAGIC      household_key INT,   
# MAGIC      day DATE,   
# MAGIC      commodity_desc STRING,   
# MAGIC      prediction DOUBLE);
# MAGIC      
# MAGIC -- truncate table and insert new records
# MAGIC INSERT OVERWRITE household_commodity_propensities__UNPIVOTED
# MAGIC SELECT * FROM TEMP__household_commodity_propensities__UNPIVOTED;
# MAGIC
# MAGIC -- display results
# MAGIC SELECT *
# MAGIC FROM household_commodity_propensities__UNPIVOTED;

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the [Databricks License](https://databricks.com/db-license-source).  All included or referenced third party libraries are subject to the licenses set forth below.

# Databricks notebook source
# MAGIC %md The purpose of this notebook is to initialize the features and models required for the propensity scoring solution accelerator. This notebook is available at https://github.com/databricks-industry-solutions/propensity-workflows.

# COMMAND ----------

# MAGIC %md ##Introduction
# MAGIC
# MAGIC With our data in place, we can now setup the environment to enable our workflows.  The resources we need to setup are:
# MAGIC </p>
# MAGIC
# MAGIC 1. Product Groupings to Score
# MAGIC 2. Historical Feature Sets
# MAGIC 3. Initial Set of Models

# COMMAND ----------

# DBTITLE 1,Get Config Settings
# MAGIC %run ./util/config

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
import pyspark.sql.functions as fn

from datetime import timedelta

# COMMAND ----------

# MAGIC %md ##Step 1: Identify Product Groupings to Score
# MAGIC
# MAGIC The products in this dataset are divided into *commodities* which serve as a kind of category assignment. Propensity scoring exercises are not necessarily aligned with whole categories and quite often may cross category boundaries (such as when we wish to promote a particular manufacturer).  But in the absence of a specific business directive, we might simply use these commodity assignments as the basis for grouping products for propensity scoring.
# MAGIC
# MAGIC There are 308 commodities in the dataset, some of which customers rarely purchase from.  Instead of generating features, models and scores for every commodity, we might define a set of commodities of interest for our marketing efforts.  We'll do this here by selecting our top 10 product categories based on number of purchases associated with each. While this works for a demonstration, you will typically want an approach informed by marketing team priorities for determining which bundle of products should be tackled within this workflow: 

# COMMAND ----------

# DBTITLE 1,Identify 10 Most Popular Commodities
top_commodities = (
  spark
    .table('transactions_adj')
    .join(spark.table('products'), on='product_id')
    .select('commodity_desc','basket_id')
    .groupBy('commodity_desc')
      .agg(fn.countDistinct('basket_id').alias('purchases'))
    .orderBy('purchases', ascending=False)
    .limit(10)
  )

display(top_commodities)

# COMMAND ----------

# MAGIC %md We'll persist these to a table that will be used in our workflows to guide their activity.  Because we will create models and define columns based on these commodity names, we'll cleanup the names a bit to align with naming requirements for the objects we intend to create:

# COMMAND ----------

# DBTITLE 1,Persist Selected Commodities to Control Workflows
_ = (
  top_commodities
    .select('commodity_desc')
    .withColumn('commodity_clean', fn.expr("regexp_replace(commodity_desc, '[-|\\/:;,.\"'']', '_')"))
    .withColumn('commodity_clean', fn.expr("replace(commodity_clean, ' ', '_')"))
    .write
      .format('delta')
      .mode('overwrite')
      .option('overwriteSchema','true')
      .saveAsTable('commodities_to_score')
  ) 

display(spark.table('commodities_to_score'))

# COMMAND ----------

# MAGIC %md ##Step 2: Generate Features for Historical Periods
# MAGIC
# MAGIC Our model training efforts will depend on the availability of features calculated as-of 30-days prior to current date.  We can imagine our workflow only calculating the latest features for the current day and accumulating feature values over time sufficient for us to perform on-going training efforts.  But because we are initializing a historical dataset in this notebook, we will need to create the features for these prior periods now:
# MAGIC
# MAGIC **NOTE** This step takes a while to complete because you are calculating features for each day in 30-day window.  You can shorten this for the purposes of this demonstration by restricting calculations to the first and last days in the range.

# COMMAND ----------

# DBTITLE 1,Calculate Features for Relevant Prior Periods
# get current day as determined by intro_and_config notebook
current_day = config['current_day']

# for days in past ...
for day in [current_day, current_day-timedelta(days=30)]:
#for d in range(30,0,-1): # for all dates in range

  ## calculate date as current day minus days prior
  #day = current_day - timedelta(days=d) # for all dates in range

  print(f"Generating features for {day}")
  dbutils.notebook.run(
    path='./04a_Task__Feature_Engineering', # notebook to run
    timeout_seconds=0, # no timeout
    arguments={
      'current day': day.strftime('%Y-%m-%d'), 
      'database name': config['database_name']
      } 
    )

# COMMAND ----------

# MAGIC %md While we have three sets of features that are created in the *Task__Feature_Engineering* notebook, we can examine one set to just confirm we have features in place:

# COMMAND ----------

# DBTITLE 1,Verify Features by Date
display(
  spark
    .table('household_features')
    .groupBy('day')
      .agg( fn.count('*').alias('records'))
    .orderBy('day')
  )

# COMMAND ----------

# MAGIC %md ##Step 3: Train Initial Models
# MAGIC
# MAGIC With our features in place, we can now train the models our daily workflow will need to generate propensity scores:
# MAGIC
# MAGIC **NOTE** This step will take a bit of time, depending on how many models you intend to create.

# COMMAND ----------

# DBTITLE 1,Train Models
dbutils.notebook.run(
  path='./04b_Task__Model_Training', # notebook to run
  timeout_seconds=0, # no timeout
  arguments={
    'current day': config['current_day'].strftime('%Y-%m-%d'), 
    'database name': config['database_name']
    } 
  )

# COMMAND ----------

# MAGIC %md ## Step 4: Setup Propensity Scores Table
# MAGIC
# MAGIC Next, we need to setup an empty table into which our propensity scores will land.  The initial table can have a bare-bones structure and no data.  During the propensity scoring task, the structure of the table will be automatically adjusted to accommodate the data to be loaded into it:
# MAGIC
# MAGIC **NOTE** We are setting up two tables to illustrate different ways of generating output.  Different systems might have different preferences for the table structure.

# COMMAND ----------

# DBTITLE 1,Setup Empty Table for Propensity Scores (Pivoted)
# MAGIC %sql
# MAGIC
# MAGIC CREATE TABLE IF NOT EXISTS household_commodity_propensities__PIVOTED
# MAGIC AS
# MAGIC   SELECT household_key, day FROM household_features WHERE 1=2;

# COMMAND ----------

# DBTITLE 1,Setup Empty Table for Propensity Scores (Unpivoted)
# MAGIC %sql CREATE TABLE IF NOT EXISTS household_commodity_propensities__UNPIVOTED
# MAGIC AS
# MAGIC     SELECT household_key, day, commodity_desc, cast(0.0 as double) as prediction
# MAGIC     FROM household_commodity_features     
# MAGIC     WHERE 1=2; 

# COMMAND ----------

# MAGIC %md ##(OPTIONAL) Step 4: Calculate Initial Propensity Scores
# MAGIC
# MAGIC At this point, we have everything in place to calculate our propensity scores.  If we want to verify our logic is working properly, we can manually call the propensity scoring task and examine the resulting output:

# COMMAND ----------

# DBTITLE 1,Calculate Propensities
dbutils.notebook.run(
  path='./04c_Task__Propensity_Estimation', # notebook to run
  timeout_seconds=0, # no timeout
  arguments={
    'current day': config['current_day'].strftime('%Y-%m-%d'), 
    'database name': config['database_name']
    } 
  )

# COMMAND ----------

# DBTITLE 1,Review Propensity Scores (Pivoted)
# MAGIC %sql
# MAGIC
# MAGIC SELECT *
# MAGIC FROM household_commodity_propensities__PIVOTED
# MAGIC LIMIT 100;

# COMMAND ----------

# DBTITLE 1,Review Propensity Scores (Unpivoted)
# MAGIC %sql
# MAGIC
# MAGIC SELECT *
# MAGIC FROM household_commodity_propensities__UNPIVOTED
# MAGIC LIMIT 100;

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the [Databricks License](https://databricks.com/db-license-source).  All included or referenced third party libraries are subject to the licenses set forth below.

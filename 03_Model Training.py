# Databricks notebook source
# MAGIC %md The purpose of this notebook is to train a propensity scoring model. 

# COMMAND ----------

# MAGIC %md ## Introduction
# MAGIC 
# MAGIC In this notebook, we'll calculate the labels associated with a set of features in order to train a model capable of calculating household propensity for a given product commodity (category). We will make use of [Databricks AutoML](https://docs.databricks.com/applications/machine-learning/automl.html) to train this model as this is help to automate much of the work associated with this and automatically employ industry best-practices for this kind of modeling effort.  Once we have a model trained, we will persiste it for scoring in the next notebook.

# COMMAND ----------

# DBTITLE 1,Retrieve Configuration Values
# MAGIC %run "./00_Overview & Configuration"

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
from databricks.feature_store import FeatureStoreClient, FeatureLookup

import mlflow
import databricks.automl

import pyspark.sql.functions as f

# COMMAND ----------

# DBTITLE 1,Set Current Database
spark.catalog.setCurrentDatabase(config['database'])

# COMMAND ----------

# MAGIC %md ## Step 1: Derive Labels
# MAGIC 
# MAGIC Our first step is to calculate the labels we will use to for model training.  As our goal is to predict whether a household will purchase products within a given product commodity within the next 30-days, we will calculate a label of 1 or 0 indicating whether or not a household purchased a product in a commodity category within the last 30-days of our dataset.
# MAGIC 
# MAGIC To do that, we will first need to assemble a dataset of all commodities intersected with all households. We'll then need to retrieve the transactional data associated with the period for which we wish to predict and identify which of those household-commodity combinations shows up in that data to then assign our label values:
# MAGIC 
# MAGIC **NOTE** Please notice that to select the transactions that should be used to derive our labels, we identify the last point in the transaction history from which features associated with these labels should be derived. In otherwords, if our labels are derived from a 30-day period, our features should be derived from day one-day or more prior to the start of that 30-day period.  That last day associated with our features was recorded in the feature set as the *day* field.  That *day* value will be included in our label set to facilitate a join to our feature set in later steps.

# COMMAND ----------

# DBTITLE 1,Retrieve Transactional Data
transactions = (
    spark
      .table('transactions_adj')
      .join(
        spark.table('products'),
        on='product_id',
        how='inner'
        )
    )

# COMMAND ----------

# DBTITLE 1,Assemble Set of All Household-Commodity Combinations
# get unique commodities
commodities = (
  spark
    .table('products')
    .select('commodity_desc')
    .distinct()
  )

# get unique households
households = transactions.select('household_key').distinct()

# cross join all commodities and households
households_commodities = households.crossJoin(commodities)

# COMMAND ----------

# DBTITLE 1,Identify Cutoffs for Features
features_day = (
  transactions
    .groupBy()
      .agg(f.max('day').alias('last_day')) # get last day for relevant feature set generation
    .selectExpr(
      'last_day - 30 as features_day'
      ) 
    .collect()
  )[0]['features_day']

print(f"Features are cutoff as of {features_day.strftime('%Y-%m-%d')}")

# COMMAND ----------

# DBTITLE 1,Retrieve Data for Label Generation
label_transactions = transactions.filter(f.expr(f"day  > '{features_day.strftime('%Y-%m-%d')}'"))

# COMMAND ----------

# DBTITLE 1,Generate Labels
# calculate label of 1 if commodity found in curent transaction set for a given household
purchased_commodities = (
  label_transactions
    .select(
      'household_key',
      'commodity_desc'
      )
    .distinct()
    .withColumn('label',f.lit(1)) # assign label=1 if in current transaction set
    )

# combine with full household-commodity set to identify commodities not purchased
training_labels = (
  households_commodities
    .join(
      purchased_commodities,
      on=['household_key','commodity_desc'],
      how='leftouter'
      )
    .withColumn('label',f.coalesce('label',f.lit(0))) # assign label=0 if not in current transaction set
    .withColumn('day', f.lit(features_day)) # identify the features to which this set should associate
  )

# display results
display(
  training_labels
    .orderBy('household_key','commodity_desc')
  )

# COMMAND ----------

# MAGIC %md We now have the complete set of labels with which we might train a model, and it's at this point we need to make a decision regarding how to proceed.  We can either train a single model to predict propensities across all 300+ commodities or we can train a model for each commodity of interest. Because we are typically only going to be interested in a small number of commodities for which we will be running promotional offers or delivering targeted messaging, we'll elect the latter for this exercise.  To help us with commodity selection, we'll implement a [widget](https://docs.databricks.com/notebooks/widgets.html) with which we can filter our label set:

# COMMAND ----------

# DBTITLE 1,Create Widget Parameter
# MAGIC %sql
# MAGIC 
# MAGIC CREATE WIDGET DROPDOWN Commodity DEFAULT "SOFT DRINKS" 
# MAGIC CHOICES 
# MAGIC   SELECT 
# MAGIC     commodity_desc
# MAGIC   FROM 
# MAGIC   (
# MAGIC     SELECT 
# MAGIC       commodity_desc, 
# MAGIC       count(commodity_desc) as count
# MAGIC     FROM products
# MAGIC     GROUP BY commodity_desc
# MAGIC     ORDER BY count desc
# MAGIC   )

# COMMAND ----------

# MAGIC %md **Note**: We will access the widget value in the same language it was defined in to avoid any confusion by the runtime for `Run All` and `Workflow` executions.

# COMMAND ----------

# DBTITLE 1,Get Widget Selection
commodity_desc = spark.sql("SELECT getArgument('Commodity')").take(1)[0][0]
commodity_desc

# COMMAND ----------

# DBTITLE 1,Filter Labels Based on Selection
training_labels = training_labels.filter(f"commodity_desc='{commodity_desc}'") 

# COMMAND ----------

# MAGIC %md ## Step 2: Retrieve Features
# MAGIC 
# MAGIC To retrieve our features, we first need to identify the fields that will be used to locate relevant feature records.  These fields - *household_key*, *commodity_desc* and *day* in this scenario - must be present in the label set we generated in order to match labels to features:

# COMMAND ----------

# DBTITLE 1,Identify Keys with which to Lookup Features
# primary key matching columns
lookup_keys = ['household_key','commodity_desc', 'day']

# COMMAND ----------

# MAGIC %md The matching of labels to features is defined through the definition of a [training_set](https://docs.databricks.com/applications/machine-learning/feature-store/concepts.html#training-set).  A training set includes instructions on how to locate features (known as a *feature lookup*) as well as details on which field represents the label and which (if any) fields to exclude from model training.  We can define our training set as follows:

# COMMAND ----------

# DBTITLE 1,Create Training Set
# instantiate feature store client
fs = FeatureStoreClient()

# define features to lookup
feature_lookups = [
  FeatureLookup(
    table_name = '{0}.propensity_features'.format(config['database']), # get data from this feature store table
    lookup_key = lookup_keys, # features looked up based on these keys (any other fields are treated as features)
    #feature_names = feature_columns # if not specified then all other columns are treated as features
    )
  ]

# combine features with labels to form a training set
training_set = fs.create_training_set(
  df = training_labels,
  feature_lookups = feature_lookups,
  label = 'label',
  exclude_columns = lookup_keys # dont train on feature lookup keys
  )

# COMMAND ----------

# MAGIC %md The training set represents a set of metadata-driven instructions with which data can be retrieved for model training purposes.  To retrieve that data, we can extract a dataframe as follows:

# COMMAND ----------

# DBTITLE 1,Extract Dataframe from Training Set
# load features and labels to a dataframe
training_df = training_set.load_df()

display(training_df)

# COMMAND ----------

# MAGIC %md ## Step 3: Perform AutoML Model Training
# MAGIC 
# MAGIC With our training set in hand, we can now train our model.  AutoML makes the steps for this pretty simple, but there is one *gotcha*.  As part of AutoML's preprocessing steps, it inspects our data and attempts to discern between categorical and continous features. To avoid confusion, we will apply explicit control over this step by applying a [semantic type annocation](https://docs.databricks.com/applications/machine-learning/automl.html#semantic-type-annotations) to each of our fields.  
# MAGIC 
# MAGIC While these annotations support *categorical*, *text*, *date* and *numeric* (continous) designations, all our feature fields are *numeric*:

# COMMAND ----------

# DBTITLE 1,Identify Features as Continuous, Numeric Data
# for each field in the training  feature set ...
for fld in training_df.schema.fields:
    
  # get the field's metadata
  meta_dict = fld.metadata

  # set its semantic type to numeric
  meta_dict['spark.contentAnnotation.semanticType'] = 'numeric'

  # persist updated metadata to dataframe
  training_df = (
    training_df
      .withMetadata(
        fld.name, 
        meta_dict
        )
    )

# COMMAND ----------

# MAGIC %md With our data prepared, we can now turn AutoML loose on the data to train a classification model.  As of the time this notebook was developed, AutoML supports the use of various algorithms within the sklearn, lightgbm and xgboost frameworks in the creation of a classification model. It leverages hyperopt to distribute the hyperparameter tuning training runs used to optimize each of these and captures model training results to an mlflow experiment. Various settings in the [API](https://docs.databricks.com/applications/machine-learning/automl.html#classification) allow us to control how these features are used.  For our purposes, we'll call AutoML as follows:
# MAGIC 
# MAGIC **NOTE** Sample output for AutoML is provided as a screenshot to avoid confusion about links to resources generated by AutoML.

# COMMAND ----------

# DBTITLE 1,Identify Best Model with AutoML
summary = databricks.automl.classify(
  training_df, # dataset containing features and labels
  target_col='label', # label column
  primary_metric='roc_auc', # metric to optimize for
  data_dir=config['automl_data_dir'], # directory for automl-generated files
  timeout_minutes=30 # maximum minutes for operation
  )

# COMMAND ----------

# DBTITLE 1,Sample AutoML Output
# MAGIC %md <img src='https://brysmiwasb.blob.core.windows.net/demos/images/propensity_automl_output.PNG' width=800>

# COMMAND ----------

# MAGIC %md The output generated by AutoML provide links to various notebook assets created during its run.  While each of these is interesting to explore, you can navigate to the MLflow experiment generated by AutoML to locate all of the assets associated with the numerous models trained during its run.
# MAGIC 
# MAGIC To locate the experiment, you can click the *navigate to the MLflow experiment* link provided in the AutoML output.  Alternatively, you can click on the *Experiments* item in the Databricks workspace UI and click on the most recent experiment named after the current notebook.  Please note that if AutoML is run multiple times from within the same notebook, you will have multiple experiments with similar names.  If this is the case, be sure to leverage the date and time information in the Experiments screen to locate the one associated with the appropriate AutoML run: </p>
# MAGIC 
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/propensity_automl_experiments.png'>

# COMMAND ----------

# MAGIC %md Once you've identified the experiment of interest, you can click on it to see details about all the different models trained during the run.  Each will be presented with various evaluation metrics that can be used to assist you in locating your preferred model: </p>
# MAGIC 
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/propensity_automl_model_iterations.png'>

# COMMAND ----------

# MAGIC %md To explore a specific model, you can click on it from within the list provided by the experiment. This will allow you to review the various evaluation metrics associated with the model and with assets created through its training:
# MAGIC </p>
# MAGIC 
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/propensity_automl_model_artifacts.PNG'>

# COMMAND ----------

# MAGIC %md To learn more about navigating the mlflow experiment UI including how to compare multiple models, please review [this document](https://docs.databricks.com/applications/mlflow/tracking.html).

# COMMAND ----------

# MAGIC %md ## Step 4: Register the Preferred Model
# MAGIC 
# MAGIC AutoML is *glass-box* solution for model training in that it preserves the notebook used to produce each model iteration.  By clicking on the *Source* notebook at the top of a specific model's page, you can see all the details that went into data preparation, model training and evaluation:
# MAGIC </p>
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/propensity_automl_model_artifacts_notebook2.PNG'>

# COMMAND ----------

# MAGIC %md If you scroll to the bottom of the notebook, you should be able to locate the *model_uri* associated with this particular model run.  Be sure to copy the complete URI, *i.e.* *runs:/.../model* before proceeding to the next notebook.
# MAGIC 
# MAGIC </p>
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/propensity_automl_run_id.png' width=1000>
# MAGIC 
# MAGIC In the block below, we programmatically select the model from the best trial as our preferred model for illustration purposes. 

# COMMAND ----------

# DBTITLE 1,Set the Model URI
model_uri = f'runs:/{summary.best_trial.mlflow_run_id}/model' # or your preferred model uri from the interactive model exploration
model_uri

# COMMAND ----------

# MAGIC %md The model URI provides us the means with which to retrieve a specific model from the mlflow repository where the AutoML-generated models reside. Our goal is now to register our preferred model along with information about the feature store assets from which it consumes features within the model registry.  This will make data scoring in our next step much easier.
# MAGIC 
# MAGIC Our first step in this is to decide on a friendly name for our model.  We'll use our selected commodity as part of this name:

# COMMAND ----------

# DBTITLE 1,Assign Model Name
model_name = 'propensity__'+commodity_desc.replace(' ','_')
model_name

# COMMAND ----------

# MAGIC %md Next, we want to ensure our model returns the probability with which a household will make a purchase from a commodity and not the default 0 or 1 label the model returns based on an internal evaluation of that probability.  To do this, we will need to retrieve our model from mlflow and persist it with a custom wrapper that overrides its default *predict* behavior:

# COMMAND ----------

# DBTITLE 1,Override Predict Method on Model
# load existing model
model = mlflow.pyfunc.load_model(model_uri)

# define custom wrapper
class modelWrapper(mlflow.pyfunc.PythonModel):
  
  def __init__(self, model):
    self.model = model
    
  def predict(self, context, model_input):
    return self.model.predict_proba(model_input)[:,1]
  
  
# wrap original model in custom wrapper
wrapped_model = modelWrapper(model._model_impl)  # _model_impl provides access to the underlying model and its full API

# COMMAND ----------

# MAGIC %md Finally, we can persist our model to the mlflow registry.  There are several ways we can do this but our preferred way here is to log the model through the Databricks file store client.
# MAGIC 
# MAGIC By logging the model through the file store client, we have the ability to capture additional metadata about how our model retrieves features.  This is done by passing the training set defined earlier to the model logging command:

# COMMAND ----------

# DBTITLE 1,Register Model with Link to Feature Store
fs.log_model(
  model=wrapped_model,
  artifact_path='model', 
  flavor=mlflow.pyfunc, 
  training_set=training_set, # log with information on how to lookup features in feature store
  registered_model_name=model_name
  )

# COMMAND ----------

# MAGIC %md We've registered our model to the mlflow registry and now need to elevate it to be the most recent production instance.  In a real-world scenario, there would typically be a review workflow and only following approvals would this model be elevated into production, often through the mlflow registry UI.  Here, we are doing this via code to facilitate our demonstration:

# COMMAND ----------

# DBTITLE 1,Elevate Newly Registered Model to Production
# connect to mlflow
client = mlflow.tracking.MlflowClient()

# identify model version in registry
model_version = client.search_model_versions(f"name='{model_name}'")[0].version

# move model version to production
client.transition_model_version_stage(
  name=model_name,
  version=model_version,
  stage='production'
  )      

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the [Databricks License](https://databricks.com/db-license-source).  All included or referenced third party libraries are subject to the licenses set forth below.

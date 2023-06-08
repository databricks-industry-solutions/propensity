# Databricks notebook source
# MAGIC %md The purpose of this notebook is to train the models required for our propensity scoring work. This notebook was developed using the **Databricks 12.2 LTS ML** runtime.

# COMMAND ----------

# MAGIC %md ##Introduction
# MAGIC
# MAGIC In this notebook, we will provide the logic needed to retrain the models for each of our product commodity (categories).  For each commodity, we will tune the model before training a final instance that will be immediately elevated to be the production instance of the propensity model for that category.
# MAGIC
# MAGIC **NOTE** Before running this notebook, make sure you have populated the feature store with features from 30 days back from the *current day*.

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
import mlflow
from hyperopt import hp, fmin, tpe, SparkTrials, STATUS_OK, space_eval

import databricks.feature_store as feature_store
from databricks.feature_store import FeatureStoreClient

from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, balanced_accuracy_score, matthews_corrcoef
from sklearn.utils.class_weight import compute_class_weight

from xgboost import XGBClassifier

import pyspark.sql.functions as fn

from datetime import datetime, timedelta
import pathlib

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

# MAGIC %md ##Step 2: Determine Date Ranges
# MAGIC
# MAGIC With the *current day* known, we now can retrieve features and derive labels. The current day is important as this represents the latest point from which we can train a model.  In our propensity scoring scenario, we envision making a prediction for likelihood to purchase over the next 30 days.  To train a model for this, we must derive a label using data 30-days back and up to the current day.  Features used to then predict that label must be derived from data prior to this.  We might understand the relationship between the current days and days prior during model training as follows:
# MAGIC </p>
# MAGIC
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/aiq_days_prior3.png' width=60%>
# MAGIC
# MAGIC </p>
# MAGIC With this in mind, we might define the start and end of our label and feature inputs as follows:

# COMMAND ----------

# DBTITLE 1,Define Cutoff Days for Features and Labels
labels_end_day = current_day
features_end_day = labels_end_day - timedelta(days=(30))

labels_start_day = features_end_day + timedelta(days=1)

print(f"We will derive features from the start of the dataset through {features_end_day}.")
print(f"We will drive labels from {labels_start_day} through {labels_end_day}, i.e. {(labels_end_day - labels_start_day).days + 1} days")

# COMMAND ----------

# MAGIC %md ##Step 2: Assemble Labels
# MAGIC
# MAGIC With the date ranges for labels defined, we can now derive labels for each household and commodity by first retrieving commodities with purchases by a given household within our label creation period.  These will be our positive class labels:

# COMMAND ----------

# DBTITLE 1,Identify Household-Commodity Pairs Positive in Label Period
positive_labels = (
    spark
      .table('transactions_adj')
      .filter(fn.expr(f"day BETWEEN '{labels_start_day}' AND '{labels_end_day}'")) # in label period
      .join(
        spark.table('products'), # join to products to get commodity assignment
        on='product_id',
        how='inner'
        )
      .join( # limit to commodities of interest
        spark.table('commodities_to_score'),
        on='commodity_desc',
        how='inner'
        )
      .select('household_key','commodity_desc') # households and commodities that saw a purchase in period
      .distinct()
      .withColumn('purchased', fn.lit(1)) # these are the positive labels
    )

display(positive_labels)

# COMMAND ----------

# MAGIC %md We can then grab ever household-commodity combination we could likely see in this same period:

# COMMAND ----------

# DBTITLE 1,Identify All Household-Commodity Combinations in Dataset
# get commodities of interest
commodities_to_score = (
  spark
    .table('commodities_to_score')
  )

# get unique households
households = spark.table('transactions_adj').select('household_key').distinct()

# cross join all commodities and households
household_commodity = households.crossJoin(commodities_to_score.select('commodity_desc'))

# COMMAND ----------

# MAGIC %md Combining these with a left-outer join, we can flag those that received a purchase with a label of 1 and those that did not with a label of 0:

# COMMAND ----------

# DBTITLE 1,Combine with Positive Labels to Determine Negative Labels
labels = (
  household_commodity
    .join(
      positive_labels, 
      on=['household_key','commodity_desc'], 
      how='leftouter'
      )
    .withColumn('day', fn.lit(features_end_day)) # day for linking to features
    .withColumn('purchased',fn.expr("coalesce(purchased, 0)"))
    .orderBy('household_key','commodity_desc')
  ).cache()

display(labels)

# COMMAND ----------

# MAGIC %md As we consider the positive and negative class labels, it can be helpful to examine the ratio of positive class instances associated with each commodity.  As is typical in most propensity scoring scenarios, we will expect to have a class imbalance within which there are very few positive instances relative to negative instances associated with each commodity:

# COMMAND ----------

# DBTITLE 1,Calculate Positive Class Ratios for Each Commodity
pos_class_ratios = (
  labels
    .groupBy('commodity_desc','purchased')
      .agg(fn.count('*').alias('class_rows'))
    .withColumn('commodity_rows', fn.expr('sum(class_rows) over(partition by commodity_desc)'))
    .filter('purchased=1')
    .withColumn('pos_class_ratio',fn.expr('class_rows/commodity_rows'))
    .select('commodity_desc','pos_class_ratio')
    ).cache()

display(pos_class_ratios)

# COMMAND ----------

# MAGIC %md ##Step 3: Retrieve Features
# MAGIC
# MAGIC We can now retrieve our features as they existed the day prior to the start of our label calculation period. Because these features were previously calculated and retained in the feature store, we can retrieve them as follows:

# COMMAND ----------

# DBTITLE 1,Define Feature Retrieval Logic (Feature Lookups)
feature_lookups = [
  # household features
  feature_store.FeatureLookup(
    table_name = f'{database_name}.household_features',
    lookup_key = ['household_key','day'],
    feature_names = [c for c in spark.table('household_features').drop('household_key','day').columns],
    rename_outputs = {c:f'household__{c}' for c in spark.table('household_features').columns}
    ),
  # commodity features
  feature_store.FeatureLookup(
    table_name = f'{database_name}.commodity_features',
    lookup_key = ['commodity_desc','day'],
    feature_names = [c for c in spark.table('commodity_features').drop('commodity_desc','day').columns],
    rename_outputs = {c:f'commodity__{c}' for c in spark.table('commodity_features').columns}
    ),
  # household-commodity features
  feature_store.FeatureLookup(
    table_name = f'{database_name}.household_commodity_features',
    lookup_key = ['household_key','commodity_desc','day'],
    feature_names = [c for c in spark.table('household_commodity_features').drop('household_key','commodity_desc','day').columns],
    rename_outputs = {c:f'household_commodity__{c}' for c in spark.table('household_commodity_features').columns}
    )
  ]

# COMMAND ----------

# MAGIC %md ##Step 4: Define Model Training Functions
# MAGIC
# MAGIC With our feature and label data retrieved, we can now launch the process to train our models.  For each model, we'll perform a hyperparameter tuning run followed by a final model training cycle.  The logic for performing a hyperparameter tuning run will be defined as follows, where the metric that is to serve as the focus of our model tuning exercise is returned as a loss value that we seek to minimize:

# COMMAND ----------

# DBTITLE 1,Define Function to Train Model Given a Set of Hyperparameter Values
def evaluate_model (hyperopt_params):
  
  # accesss replicated input data
  _X_train = X_train_broadcast.value
  _y_train = y_train_broadcast.value
  _X_validate = X_validate_broadcast.value
  _y_validate = y_validate_broadcast.value
  
  # configure model parameters
  params = hyperopt_params
  if 'max_depth' in params: params['max_depth']=int(params['max_depth'])   # hyperopt supplies values as float but must be int
  if 'min_child_weight' in params: params['min_child_weight']=int(params['min_child_weight']) # hyperopt supplies values as float but must be int
  if 'max_delta_step' in params: params['max_delta_step']=int(params['max_delta_step']) # hyperopt supplies values as float but must be int
  # all other hyperparameters are taken as given by hyperopt
  
  # instantiate model with parameters
  model = XGBClassifier(**params)
  
  # train
  model.fit(X_train, y_train)
  
  # predict
  y_pred = model.predict(X_validate)
  y_prob = model.predict_proba(X_validate)
  
  # eval metrics
  model_ap = average_precision_score(y_validate, y_prob[:,1])
  model_ba = balanced_accuracy_score(y_validate, y_pred)
  model_mc = matthews_corrcoef(y_validate, y_pred)
  
  # log metrics with mlflow run
  mlflow.log_metrics({
    'avg precision':model_ap,
    'balanced_accuracy':model_ba,
    'matthews corrcoef':model_mc
    })                                       
                                             
  # invert key metric for hyperopt
  loss = -1 * model_ap
  
  # return results
  return {'loss': loss, 'status': STATUS_OK}

# COMMAND ----------

# MAGIC %md Similarly, we can define a function to train our model based on the discovered set of *best* hyperparameter values.  Note that this function's logic closely mirrors that of the function used for hyperparameter tuning with the exception that our return values differ and we intend to train this final model on our cluster's driver and not on the worker nodes (so that we do not need to mess with broadcasted datasets):

# COMMAND ----------

# DBTITLE 1,Define Function to Train Model Given Best Hyperparameter Values
def train_final_model (hyperopt_params):
   
  # configure model parameters
  params = hyperopt_params
  if 'max_depth' in params: params['max_depth']=int(params['max_depth'])   # hyperopt supplies values as float but must be int
  if 'min_child_weight' in params: params['min_child_weight']=int(params['min_child_weight']) # hyperopt supplies values as float but must be int
  if 'max_delta_step' in params: params['max_delta_step']=int(params['max_delta_step']) # hyperopt supplies values as float but must be int
  # all other hyperparameters are taken as given by hyperopt
  
  # instantiate model with parameters
  model = XGBClassifier(**params)
  
  # train
  model.fit(X_train_validate, y_train_validate)
  
  # predict
  y_pred = model.predict(X_test)
  y_prob = model.predict_proba(X_test)
  
  # eval metrics
  model_ap = average_precision_score(y_test, y_prob[:,1])
  model_ba = balanced_accuracy_score(y_test, y_pred)
  model_mc = matthews_corrcoef(y_test, y_pred)

  scores = {
    'avg precision':model_ap,
    'balanced_accuracy':model_ba,
    'matthews corrcoef':model_mc
    }
  
  # log metrics with mlflow run
  mlflow.log_metrics(scores)

  return model, scores

# COMMAND ----------

# MAGIC %md ##Step 5: Train Per-Commodity Models
# MAGIC
# MAGIC We can now tune and train models for each of the commodities we wish to score:
# MAGIC
# MAGIC **NOTE** Because models cannot be logged to notebook-aligned experiments when run as part of a workflow, we are explicitly setting an independent experiment here.

# COMMAND ----------

# DBTITLE 1,Set Experiment within Which to Capture Outputs
username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
mlflow.set_experiment('/Users/{}/propensity'.format(username))

# COMMAND ----------

# DBTITLE 1,Retrieve Commodities to Score
commodities = (
  commodities_to_score
  ).toPandas().to_dict(orient='records')

# COMMAND ----------

# DBTITLE 1,Tune & Train Models for Each Commodity
# for each commodity_desc
for commodity in commodities:

  commodity_desc = commodity['commodity_desc']

  print(commodity_desc)

  # model name can't have /,: or . chars
  model_name = f"propensity {commodity['commodity_clean']}"

  # instantiate feature store client
  fs = FeatureStoreClient()

  # ASSEMBLE FEATURES AND LABELS
  # --------------------------------------------------------
  # get training set
  training_set = fs.create_training_set(
      labels.filter(f"commodity_desc='{commodity_desc}'"),
      feature_lookups=feature_lookups,
      label='purchased',
      exclude_columns=['household_key','commodity_desc','day']
      )

  # get features and labels
  features_and_labels = training_set.load_df().toPandas()
  X = features_and_labels.drop('purchased', axis=1)
  y = features_and_labels['purchased']

  # split into train (0.70), validate (0.15) and test (0.15)
  X_train_validate, X_test,  y_train_validate, y_test = train_test_split(X, y, test_size=0.15)
  X_train, X_validate, y_train, y_validate = train_test_split(X_train_validate, y_train_validate, test_size=(0.15/0.85))

  # broadcast sets
  X_train_broadcast = sc.broadcast(X_train)
  y_train_broadcast = sc.broadcast(y_train)
  X_validate_broadcast = sc.broadcast(X_validate)
  y_validate_broadcast = sc.broadcast(y_validate)
  # --------------------------------------------------------

  # PERFORM HYPERPARAMETER TUNING
  # --------------------------------------------------------
  # define search space
  search_space = {
      'max_depth' : hp.quniform('max_depth', 5, 20, 1)                       
      ,'learning_rate' : hp.uniform('learning_rate', 0.01, 0.40) 
      }

  # determine pos_class_weight
  pos_class_ratio = pos_class_ratios.filter(f"commodity_desc='{commodity_desc}'").collect()[0]['pos_class_ratio']
  pos_class_weight = 1.0 / pos_class_ratio
  if pos_class_weight > 1.0:
    search_space['scale_pos_weight'] = hp.uniform('scale_pos_weight', 1.0, 5 * pos_class_weight)


  # run as many in parallel as you can
  trails = sc.defaultParallelism

  # get at least 50 trails in but ideally 5 per node
  max_evals = max(50, trails*5)

  # perform tuning
  with mlflow.start_run(run_name=f'tuning {commodity_desc}'):
    # reset argmin
    argmin = None
    # try hyperparameter tuning
    try:
      argmin = fmin(
        fn=evaluate_model,
        space=search_space,
        algo=tpe.suggest,  # algorithm controlling how hyperopt navigates the search space
        max_evals=max_evals,
        trials=SparkTrials(parallelism=trails)
        )
    except:
      pass
  # --------------------------------------------------------

  # TRAIN TUNED MODEL
  # --------------------------------------------------------
  if argmin is None:
    print(f'{commodity_desc} failed!')
  else:
    # train final model
    with mlflow.start_run(run_name=f'final {commodity_desc}'):

      model, scores = train_final_model(argmin)

      _ = fs.log_model(
        model,
        artifact_path='model',
        flavor=mlflow.sklearn,
        training_set=training_set,
        registered_model_name=model_name,
        **{
          'pyfunc_predict_fn':'predict_proba',
          'pip_requirements':['xgboost']
          }
        )

    # elevate to production
    client = mlflow.tracking.MlflowClient()
    model_version = client.search_model_versions(f"name='{model_name}'")[0].version
    client.transition_model_version_stage(
      name=model_name,
      version=model_version,
      stage='production'
      )
  # --------------------------------------------------------

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the [Databricks License](https://databricks.com/db-license-source).  All included or referenced third party libraries are subject to the licenses set forth below.

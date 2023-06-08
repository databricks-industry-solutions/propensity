# Databricks notebook source
# MAGIC %md The purpose of this notebook is to engineer the features required for our propensity scoring work. This notebook was developed using the **Databricks 12.2 LTS ML** runtime.

# COMMAND ----------

# MAGIC %md ## Introduction
# MAGIC
# MAGIC In this notebook we will leverage transactional data associated with individual households to generate features with which we will train our model and, later, perform inference, *i.e.* make predictions.  Our goal is to predict the likelihood a household will purchase products from a given product category, *i.e.* commodity-designation, in the next 30 days. 
# MAGIC
# MAGIC In an operational workflow which we can imagine running separately from this, we would receive new data into the lakehouse on a periodic, *i.e.* daily or more frequent, basis.  As that data arrives, we might recalculate new or updated features and store these for the purpose of making predictions about a future period. To train a model, we'd need the state of these features some period of time, *i.e.* 30 days in this scenario, behind the current features. For this reason, it will be important to keep past versions of features for some limited duration, *i.e.* at least 30 days in this scenario.
# MAGIC
# MAGIC This notebook represents the logic associated with training one set of features for a single date. The *current_date* will be calculated in a separate notebooks and either accessed as part of the workflow or passed directly into this notebook via a widget.

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
from pyspark.sql.types import *
import pyspark.sql.functions as fn
from pyspark.sql.window import Window

from databricks.feature_store import FeatureStoreClient

from datetime import datetime, timedelta

# COMMAND ----------

# MAGIC %md ##Step 1: Access Data from which to Derive Features
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

# MAGIC %md Using our configuration values, we can now retrieve the transaction data from which we will derive our features:

# COMMAND ----------

# DBTITLE 1,Get Transaction Inputs Up to Current Date
transactions = (
    spark
      .table('transactions_adj')
      .join(
        spark.table('products'), # join to products to get commodity assignment
        on='product_id',
        how='inner'
        )
      .filter(fn.expr(f"day <= '{current_day}'"))
    )

display(transactions)

# COMMAND ----------

# MAGIC %md In addition to the raw transaction data, we need to assemble the full set of all households and commodities for which we may wish to derive features:

# COMMAND ----------

# DBTITLE 1,Get All Household-Commodity Combinations Possible for this Period
# get unique commodities
commodities = (
  spark
    .table('commodities_to_score')
    .select('commodity_desc')
  )

# get unique households
households = transactions.select('household_key').distinct()

# cross join all commodities and households
household_commodity = households.crossJoin(commodities)

# COMMAND ----------

# MAGIC %md ## Step 2: Define Feature Generation Logic
# MAGIC
# MAGIC The feature generation logic will be used to derive values from 30 day, 60 day, 90 day and 1 year prior to the *current_day*. A wide range of metrics will be calculated for each period, but what is captured here is by no means exhaustive of what we could derive from this dataset.  The encapsulation of this logic as a function will allow us to re-use this logic as we derive features for households, commodities and household-commodity combinations later:

# COMMAND ----------

# DBTITLE 1,Define Function to Derive Individual Feature Sets
def get_features(df, grouping_keys, window=None):
  
  '''
  This function derives a number of features from our transactional data.
  
  df: the dataframe containing household transaction history

  grouping_keys: the household_key, commodity_desc or combination househould_key & commodity_desc around which to group data
  
  window: one of four supported string values:
    '30d': derive metrics from the last 30 days of the dataset
    '60d': derive metrics from the last 60 days of the dataset
    '90d': derive metrics from the last 90 days of the dataset
    '1yr': derive metrics from the 30 day period starting 1-year
           prior to the end of the dataset. this alings with the
           period from which our labels are derived.
  '''

  # get list of distinct grouping items in the original dataframe
  anchor_df = transactions.select(grouping_keys).distinct()
  
  # identify when dataset starts and ends
  min_day, max_day = (
    df
      .groupBy()
        .agg(
          fn.min('day').alias('min_day'), 
          fn.max('day').alias('max_day')
          )
      .collect()
    )[0]    
  
  ## print info to support validation
  #print('{0}:\t{1} days in original set between {2} and {3}'.format(window, (max_day - min_day).days + 1, min_day, max_day))
  
  # adjust min and max days based on specified window   
  if window == '30d':
    window_suffix = '_'+window
    min_day = max_day - timedelta(days=30-1)
    
  elif window == '60d':
    window_suffix = '_'+window
    min_day = max_day - timedelta(days=60-1)
    
  elif window == '90d':
    window_suffix = '_'+window
    min_day = max_day - timedelta(days=90-1)
    
  elif window == '1yr':
    window_suffix = '_'+window
    min_day = max_day - timedelta(days=365-1)
    max_day = min_day + timedelta(days=30-1)
    
  else:
    raise Exception('unknown window definition')
  
  # determine the number of days in the window
  days_in_window = (max_day - min_day).days + 1
  
  ## print to help with date math validation
  #print('{0}:\t{1} days in adjusted set between {2} and {3}'.format(window, days_in_window, min_day, max_day))
  
  # convert dates to strings to make remaining steps easier
  max_day = max_day.strftime('%Y-%m-%d')
  min_day = min_day.strftime('%Y-%m-%d')
  
  # derive summary features from set
  summary_df = (
    df
      .filter(fn.expr(f"day between '{min_day}' and '{max_day}'")) # constrain to window
      .groupBy(grouping_keys)
        .agg(
          
          # summary metrics
          fn.countDistinct('day').alias('days'), 
          fn.countDistinct('basket_id').alias('baskets'),
          fn.count('product_id').alias('products'), 
          fn.count('*').alias('line_items'),
          fn.sum('amount_list').alias('amount_list'),
          fn.sum('instore_discount').alias('instore_discount'),
          fn.sum('campaign_coupon_discount').alias('campaign_coupon_discount'),
          fn.sum('manuf_coupon_discount').alias('manuf_coupon_discount'),
          fn.sum('total_coupon_discount').alias('total_coupon_discount'),
          fn.sum('amount_paid').alias('amount_paid'),
          
          # unique days with activity
          fn.countDistinct(
            fn.expr('case when instore_discount >0 then day else null end')
            ).alias('days_with_instore_discount'),
          fn.countDistinct(
            fn.expr('case when campaign_coupon_discount >0 then day else null end')
            ).alias('days_with_campaign_coupon_discount'),
          fn.countDistinct(
            fn.expr('case when manuf_coupon_discount >0 then day else null end')
            ).alias('days_with_manuf_coupon_discount'),
          fn.countDistinct(
            fn.expr('case when total_coupon_discount >0 then day else null end')
            ).alias('days_with_total_coupon_discount'),
          
          # unique baskets with activity
          fn.countDistinct(
            fn.expr('case when instore_discount >0 then basket_id else null end')
            ).alias('baskets_with_instore_discount'),
          fn.countDistinct(
            fn.expr('case when campaign_coupon_discount >0 then basket_id else null end')
            ).alias('baskets_with_campaign_coupon_discount'),
          fn.countDistinct(
            fn.expr('case when manuf_coupon_discount >0 then basket_id else null end')
            ).alias('baskets_with_manuf_coupon_discount'),
          fn.countDistinct(
            fn.expr('case when total_coupon_discount >0 then basket_id else null end')
            ).alias('baskets_with_total_coupon_discount'),          
    
          # unique products with activity
          fn.countDistinct(
            fn.expr('case when instore_discount >0 then product_id else null end')
            ).alias('products_with_instore_discount'),
          fn.countDistinct(
            fn.expr('case when campaign_coupon_discount >0 then product_id else null end')
            ).alias('products_with_campaign_coupon_discount'),
          fn.countDistinct(
            fn.expr('case when manuf_coupon_discount >0 then product_id else null end')
            ).alias('products_with_manuf_coupon_discount'),
          fn.countDistinct(
            fn.expr('case when total_coupon_discount >0 then product_id else null end')
            ).alias('products_with_total_coupon_discount'),          
    
          # unique line items with activity
          fn.sum(
            fn.expr('case when instore_discount >0 then 1 else null end')
            ).alias('line_items_with_instore_discount'),
          fn.sum(
            fn.expr('case when campaign_coupon_discount >0 then 1 else null end')
            ).alias('line_items_with_campaign_coupon_discount'),
          fn.sum(
            fn.expr('case when manuf_coupon_discount >0 then 1 else null end')
            ).alias('line_items_with_manuf_coupon_discount'),
          fn.sum(
            fn.expr('case when total_coupon_discount >0 then 1 else null end')
            ).alias('line_items_with_total_coupon_discount')          
          )    
    
      # per-day ratios
      .withColumn(
        f'baskets_per_day', 
        fn.expr('baskets/days')
        )
      .withColumn(
        f'products_per_day{window_suffix}', 
        fn.expr('products/days')
        )
      .withColumn(
        f'line_items_per_day', 
        fn.expr('line_items/days')
        )
      .withColumn(
        f'amount_list_per_day', 
        fn.expr('amount_list/days')
        )
      .withColumn(
        f'instore_discount_per_day', 
        fn.expr('instore_discount/days')
        )
      .withColumn(
        f'campaign_coupon_discount_per_day', 
        fn.expr('campaign_coupon_discount/days')
        )
      .withColumn(
        f'manuf_coupon_discount_per_day', 
        fn.expr('manuf_coupon_discount/days')
        )
      .withColumn(
        f'total_coupon_discount_per_day', 
        fn.expr('total_coupon_discount/days')
        )
      .withColumn(
        f'amount_paid_per_day', 
        fn.expr('amount_paid/days')
        )
      .withColumn(
        f'days_with_instore_discount_per_days', 
        fn.expr('days_with_instore_discount/days')
        )
      .withColumn(
        f'days_with_campaign_coupon_discount_per_days', 
        fn.expr('days_with_campaign_coupon_discount/days')
        )
      .withColumn(
        f'days_with_manuf_coupon_discount_per_days',
        fn.expr('days_with_manuf_coupon_discount/days')
        )
      .withColumn(
        f'days_with_total_coupon_discount_per_days', 
        fn.expr('days_with_total_coupon_discount/days')
        )
    
      # per-day-in-set ratios
      .withColumn(
        f'days_to_days_in_set', 
        fn.expr(f'days/{days_in_window}')
        )
      .withColumn(
        f'baskets_per_days_in_set', 
        fn.expr(f'baskets/{days_in_window}')
        )
      .withColumn(
        f'products_to_days_in_set', 
        fn.expr(f'products/{days_in_window}')
        )
      .withColumn(
        f'line_items_per_days_in_set', 
        fn.expr(f'line_items/{days_in_window}')
        )
      .withColumn(
        f'amount_list_per_days_in_set', 
        fn.expr(f'amount_list/{days_in_window}')
        )
      .withColumn(
        f'instore_discount_per_days_in_set', 
        fn.expr(f'instore_discount/{days_in_window}')
        )
      .withColumn(
        f'campaign_coupon_discount_per_days_in_set', 
        fn.expr(f'campaign_coupon_discount/{days_in_window}')
        )
      .withColumn(
        f'manuf_coupon_discount_per_days_in_set', 
        fn.expr(f'manuf_coupon_discount/{days_in_window}')
        )
      .withColumn(
        f'total_coupon_discount_per_days_in_set', 
        fn.expr(f'total_coupon_discount/{days_in_window}')
        )
      .withColumn(
        f'amount_paid_per_days_in_set', 
        fn.expr(f'amount_paid/{days_in_window}')
        )
      .withColumn(
        f'days_with_instore_discount_per_days_in_set', 
        fn.expr(f'days_with_instore_discount/{days_in_window}')
        )
      .withColumn(
        f'days_with_campaign_coupon_discount_per_days_in_set', 
        fn.expr(f'days_with_campaign_coupon_discount/{days_in_window}')
        )
      .withColumn(
        f'days_with_manuf_coupon_discount_per_days_in_set', 
        fn.expr(f'days_with_manuf_coupon_discount/{days_in_window}')
        )
      .withColumn(
        f'days_with_total_coupon_discount_per_days_in_set', 
        fn.expr(f'days_with_total_coupon_discount/{days_in_window}')
        )

      # per-basket ratios
      .withColumn(
        'products_per_basket', 
        fn.expr('products/baskets')
        )
      .withColumn(
        'line_items_per_basket', 
        fn.expr('line_items/baskets')
        )
      .withColumn(
        'amount_list_per_basket', 
        fn.expr('amount_list/baskets')
        )      
      .withColumn(
        'instore_discount_per_basket', 
        fn.expr('instore_discount/baskets')
        )  
      .withColumn(
        'campaign_coupon_discount_per_basket', 
        fn.expr('campaign_coupon_discount/baskets')
        ) 
      .withColumn(
        'manuf_coupon_discount_per_basket', 
        fn.expr('manuf_coupon_discount/baskets')
        )
      .withColumn(
        'total_coupon_discount_per_basket', 
        fn.expr('total_coupon_discount/baskets')
        )    
      .withColumn(
        'amount_paid_per_basket', 
        fn.expr('amount_paid/baskets')
        )
      .withColumn(
        'baskets_with_instore_discount_per_baskets', 
        fn.expr('baskets_with_instore_discount/baskets')
        )
      .withColumn(
        'baskets_with_campaign_coupon_discount_per_baskets', 
        fn.expr('baskets_with_campaign_coupon_discount/baskets')
        )
      .withColumn(
        'baskets_with_manuf_coupon_discount_per_baskets', 
        fn.expr('baskets_with_manuf_coupon_discount/baskets')
        )
      .withColumn(
        'baskets_with_total_coupon_discount_per_baskets', 
        fn.expr('baskets_with_total_coupon_discount/baskets')
        )
      
      # per-product ratios
      .withColumn(
        'line_items_per_product', 
        fn.expr('line_items/products')
        )
      .withColumn(
        'amount_list_per_product', 
        fn.expr('amount_list/products')
        )      
      .withColumn(
        'instore_discount_per_product', 
        fn.expr('instore_discount/products')
        )  
      .withColumn(
        'campaign_coupon_discount_per_product', 
        fn.expr('campaign_coupon_discount/products')
        ) 
      .withColumn(
        'manuf_coupon_discount_per_product', 
        fn.expr('manuf_coupon_discount/products')
        )
      .withColumn(
        'total_coupon_discount_per_product', 
        fn.expr('total_coupon_discount/products')
        )    
      .withColumn(
        'amount_paid_per_product', 
        fn.expr('amount_paid/products')
        )
      .withColumn(
        'products_with_instore_discount_per_product', 
        fn.expr('products_with_instore_discount/products')
        )
      .withColumn(
        'products_with_campaign_coupon_discount_per_product', 
        fn.expr('products_with_campaign_coupon_discount/products')
        )
      .withColumn(
        'products_with_manuf_coupon_discount_per_product', 
        fn.expr('products_with_manuf_coupon_discount/products')
        )
      .withColumn(
        'products_with_total_coupon_discount_per_product', 
        fn.expr('products_with_total_coupon_discount/products')
        )
      
      # per-line_item ratios
      .withColumn(
        'amount_list_per_line_item', 
        fn.expr('amount_list/line_items')
        )      
      .withColumn(
        'instore_discount_per_line_item', 
        fn.expr('instore_discount/line_items')
        )  
      .withColumn(
        'campaign_coupon_discount_per_line_item', 
        fn.expr('campaign_coupon_discount/line_items')
        ) 
      .withColumn(
        'manuf_coupon_discount_per_line_item', 
        fn.expr('manuf_coupon_discount/line_items')
        )
      .withColumn(
        'total_coupon_discount_per_line_item', 
        fn.expr('total_coupon_discount/line_items')
        )    
      .withColumn(
        'amount_paid_per_line_item', 
        fn.expr('amount_paid/line_items')
        )
      .withColumn(
        'products_with_instore_discount_per_line_item', 
        fn.expr('products_with_instore_discount/line_items')
        )
      .withColumn(
        'products_with_campaign_coupon_discount_per_line_item', 
        fn.expr('products_with_campaign_coupon_discount/line_items')
        )
      .withColumn(
        'products_with_manuf_coupon_discount_per_line_item', 
        fn.expr('products_with_manuf_coupon_discount/line_items')
        )
      .withColumn(
        'products_with_total_coupon_discount_per_line_item', 
        fn.expr('products_with_total_coupon_discount/line_items')
        )    
    
      # amount_list ratios
      .withColumn(
        'campaign_coupon_discount_to_amount_list', 
        fn.expr('campaign_coupon_discount/amount_list')
        )
      .withColumn(
        'manuf_coupon_discount_to_amount_list', 
        fn.expr('manuf_coupon_discount/amount_list')
        )
      .withColumn(
        'total_coupon_discount_to_amount_list', 
        fn.expr('total_coupon_discount/amount_list')
        )
      .withColumn(
        'amount_paid_to_amount_list', 
        fn.expr('amount_paid/amount_list')
        )
      )
 
  # derive days-since metrics
  dayssince_df = (
    df
      .filter(fn.expr(f"day <= '{max_day}'"))
      .groupBy(grouping_keys)
        .agg(
          fn.min(
            fn.expr(f"'{max_day}' - case when instore_discount >0 then day else '{min_day}' end")
            ).alias('days_since_instore_discount'),
          fn.min(
            fn.expr(f"'{max_day}' - case when campaign_coupon_discount >0 then day else '{min_day}' end")
            ).alias('days_since_campaign_coupon_discount'),
          fn.min(
            fn.expr(f"'{max_day}' - case when manuf_coupon_discount >0 then day else '{min_day}' end")
            ).alias('days_since_manuf_coupon_discount'),
          fn.min(
            fn.expr(f"'{max_day}' - case when total_coupon_discount >0 then day else '{min_day}' end"))
            .alias('days_since_total_coupon_discount')
          )
      )
  
  # combine metrics with anchor set to form return set 
  ret_df = (
    anchor_df
      .join(summary_df, on=grouping_keys, how='leftouter')
      .join(dayssince_df, on=grouping_keys, how='leftouter')
    )
  
  # rename fields based on control parameters
  for c in ret_df.columns:
    if c not in grouping_keys: # don't rename grouping fields
      ret_df = ret_df.withColumn(c, fn.col(c).cast(DoubleType())) # cast all metrics as doubles to avoid confusion as categoricals
      ret_df = ret_df.withColumnRenamed(c,f'{c}{window_suffix}')

  return ret_df

# COMMAND ----------

# MAGIC %md ##Step 3: Generate Household Features
# MAGIC
# MAGIC Using our transaction inputs, we can derive household-level features as follows:

# COMMAND ----------

# DBTITLE 1,Calculate Household Features
# features will be grouped on households
grouping_keys = ['household_key']

# get master set of household keys in incoming data
features = (
  household_commodity
    .select(grouping_keys)
    .distinct()
    .withColumn('day', fn.lit(current_day)) # assign date to feature set
    )

# calculate household features for each period and join to master set
for window in ['30d','60d','90d','1yr']:
  features = (
    features
      .join(
          get_features(transactions, grouping_keys, window=window), 
          on=grouping_keys, 
          how='leftouter'
        )
    )

# fill-in any missing values
household_features = features.fillna(value=0.0, subset=[c for c in features.columns if c not in grouping_keys])

# COMMAND ----------

# MAGIC %md We can now write these data to our feature store as follows.  Please note that we are using the *household_key* field in combination with the *day* field for the unique identifier for these records.  While feature store tables support a timestamp column for versioning of records (as part of the [time series feature table](https://docs.databricks.com/machine-learning/feature-store/time-series.html) capability), in practice we have found use of this feature to be very slow compared to just placing the timestamp in the primary key.  The key - no pun intended - to making this hack work is that you must have a perfect match for the timestamp value in data used to retrieve features.  The time series feature table capability allows you to retrieve the feature version available at a given point in time but does not require a perfect match:

# COMMAND ----------

# DBTITLE 1,Write Features to Feature Store
# instantiate feature store client
fs = FeatureStoreClient()

# create feature store table (we will receive a warning with each call after the table has been created)
try: # if feature store does not exist
  fs.get_table(f'{database_name}.household_features')
except: # create it now
  pass
  _ = (
    fs
      .create_table(
        name=f'{database_name}.household_features', # name of feature store table
        primary_keys= grouping_keys + ['day'], # name of keys that will be used to locate records
        schema=household_features.schema, # schema of feature set as derived from our feature_set dataframe
        description='household features used for propensity scoring' 
      )
    )

# merge feature set data into feature store
_ = (
  fs
    .write_table(
      name=f'{database_name}.household_features',
      df = household_features,
      mode = 'merge' # merge data into existing feature set, instead of 'overwrite'
    )
  )

# COMMAND ----------

# MAGIC %md We can verify our data by retrieving features from the feature table for the *current_day*:

# COMMAND ----------

# DBTITLE 1,Verify Features
display(
  fs
    .read_table(f'{database_name}.household_features')
    .filter(fn.expr(f"day='{current_day.strftime('%Y-%m-%d')}'"))
  )

# COMMAND ----------

# MAGIC %md ##Step 4: Generate Commodity Features
# MAGIC
# MAGIC We can now do the same for each commodity in the dataset:

# COMMAND ----------

# DBTITLE 1,Calculate Commodity Features
# features will be grouped on households
grouping_keys = ['commodity_desc']

# get master set of household keys in incoming data
features = (
  household_commodity
    .select(grouping_keys)
    .distinct()
    .withColumn('day', fn.lit(current_day)) # assign date to feature set
    )

# calculate household features for each period and join to master set
for window in ['30d','60d','90d','1yr']:
  features = (
    features
      .join(
          get_features(transactions, grouping_keys, window=window), 
          on=grouping_keys, 
          how='leftouter'
        )
    )

# fill-in any missing values
commodity_features = features.fillna(value=0.0, subset=[c for c in features.columns if c not in grouping_keys])

# COMMAND ----------

# DBTITLE 1,Write Features to Feature Store
# instantiate feature store client
fs = FeatureStoreClient()

# create feature store table (we will receive a warning with each call after the table has been created)
try: # if feature store does not exist
  fs.get_table(f'{database_name}.commodity_features')
except: # create it now
  pass
  _ = (
    fs
      .create_table(
        name=f'{database_name}.commodity_features', # name of feature store table
        primary_keys= grouping_keys + ['day'], # name of keys that will be used to locate records
        schema=commodity_features.schema, # schema of feature set as derived from our feature_set dataframe
        description='commodity features used for propensity scoring' 
      )
    )

# merge feature set data into feature store
_ = (
  fs
    .write_table(
      name=f'{database_name}.commodity_features',
      df = commodity_features,
      mode = 'merge' # merge data into existing feature set, instead of 'overwrite'
    )
  )

# COMMAND ----------

# DBTITLE 1,Verify Features
display(
  fs
    .read_table(f'{database_name}.commodity_features')
    .filter(fn.expr(f"day='{current_day.strftime('%Y-%m-%d')}'"))
  )

# COMMAND ----------

# MAGIC %md ##Step 5: Generate Household-Commodity Features
# MAGIC
# MAGIC And now we can tackle the household-commodity combinations as follows:

# COMMAND ----------

# DBTITLE 1,Calculate Household-Commodity Features
# features will be grouped on households & commodities
grouping_keys = ['household_key','commodity_desc']

# get master set of household & commoditiy keys in incoming data
features = (
  household_commodity
    .select(grouping_keys)
    .distinct()
    .withColumn('day', fn.lit(current_day)) # assign date to feature set
    )

# calculate household-commodity features for each period and join to master set
for window in ['30d','60d','90d','1yr']:
  features = (
    features
      .join(
          get_features(transactions, grouping_keys, window=window), 
          on=grouping_keys, 
          how='leftouter'
        )
    )

# fill-in any missing values
household_commodity_features = features.fillna(value=0.0, subset=[c for c in features.columns if c not in grouping_keys])

# COMMAND ----------

# DBTITLE 1,Write Features to Feature Store
# instantiate feature store client
fs = FeatureStoreClient()

# create feature store table (we will receive a warning with each call after the table has been created)
try: # if feature store does not exist
  fs.get_table(f'{database_name}.household_commodity_features')
except: # create it now
  pass
  _ = (
    fs
      .create_table(
        name=f'{database_name}.household_commodity_features', # name of feature store table
        primary_keys= grouping_keys + ['day'], # name of keys that will be used to locate records
        schema=household_commodity_features.schema, # schema of feature set as derived from our feature_set dataframe
        description='household-commodity features used for propensity scoring' 
      )
    )

# merge feature set data into feature store
_ = (
  fs
    .write_table(
      name=f'{database_name}.household_commodity_features',
      df = household_commodity_features,
      mode = 'merge' # merge data into existing feature set, instead of 'overwrite'
    )
  )

# COMMAND ----------

# DBTITLE 1,Verify Features
display(
  fs
    .read_table(f'{database_name}.household_commodity_features')
    .filter(fn.expr(f"day='{current_day.strftime('%Y-%m-%d')}'"))
  )

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the [Databricks License](https://databricks.com/db-license-source).  All included or referenced third party libraries are subject to the licenses set forth below.

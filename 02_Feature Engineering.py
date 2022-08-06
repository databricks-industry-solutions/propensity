# Databricks notebook source
# MAGIC %md The purpose of this notebook is to engineer the features required for our propensity scoring work. 

# COMMAND ----------

# MAGIC %md ## Introduction
# MAGIC 
# MAGIC In this notebook we will leverage transactional data associated with individual households to generate features with which we will train our model and, later, perform inference, *i.e.* make predictions.  Our goal is to predict the likelihood a household will purchase products from a given product category, *i.e.* commodity-designation, in the next 30 days. 
# MAGIC 
# MAGIC In an operationalized workflow, we would receive new data into the lakehouse on a periodic, *i.e.* daily or more frequent basis.  As that data arrives, we might recalculate features for propensity scoring and store these for the purpose of makind predictions, *i.e.* performing inference, about the future period. As these features age, they at some point become useful for training new models.  This happens at the point that enough new data arrives that we can derive labels for the period they were built to predict.
# MAGIC 
# MAGIC To simulate this workflow, we will calculate features for each of the last 30-days of our dataset. We will establish our workflow logic at the top of this notebook and then define a loop at the bottom to persist these data for later use.  As part of this, we will be persisting our data to the [Databricks Feature Store](https://docs.databricks.com/applications/machine-learning/feature-store/index.html#), a capability in the Databricks platform which simplifies the persistence and retrieval of features. 
# MAGIC 
# MAGIC **NOTE** In this notebook, we are deriving features exclusively from our transactional sales data in order to keep things simple.  The dataset provides access to customer demographic and promotional campaign data from which additional features would typically be derived.

# COMMAND ----------

# DBTITLE 1,Retrieve Configuration Values
# MAGIC %run "./00_Overview & Configuration"

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
from pyspark.sql.types import *
import pyspark.sql.functions as f
from pyspark.sql.window import Window

from databricks.feature_store import FeatureStoreClient

from datetime import timedelta

# COMMAND ----------

# DBTITLE 1,Set Current Database
spark.catalog.setCurrentDatabase(config['database'])

# COMMAND ----------

# MAGIC %md ## Step 1: Define Feature Generation Logic
# MAGIC 
# MAGIC Our first step is to define a function to generate features from a dataframe of transactional data passed to it. In our function, we are deriving a generic set of features from the last 30, 60 and 90 day periods of the transactional data as well as from a 30-day period (aligned with the labels we wish to predict) from 1-year back.  This is not exhaustive of what we could derive from these data but should give a since of how we might approach feature generation.
# MAGIC 
# MAGIC **NOTE** If we were to derive metrics from across the entire set of data, it would be important to adust the starting date from which metrics might be derived so that features generated on different days are calculated from a consistent range of dates.
# MAGIC 
# MAGIC It's important to note that we will be deriving these features first from the household level and then from the household-commodity level.  The *include_commodity* argument is used to control which of these levels is employed at feature generation.  In later steps, these features will be combined so that each instance in our feature set will contain features for a given household as well as for that household in combination with one of the 308 commodities against which we may calculate propensity scores:

# COMMAND ----------

# DBTITLE 1,Define Function to Derive Features
def get_features(df, include_commodity=False, window=None):
  
  '''
  This function derives a number of features from our transactional data.
  These data are grouped by either just the household_key or the household_key
  and commodity_desc field and are filtered based on a window prescribed
  with the function call.
  
  df: the dataframe containing household transaction history
  
  include_commodity: controls whether data grouped on:
     household_key (include_commodity=False) or 
     household_key and commodity_desc (include_commodity=True)
  
  window: one of four supported string values:
    '30d': derive metrics from the last 30 days of the dataset
    '60d': derive metrics from the last 60 days of the dataset
    '90d': derive metrics from the last 90 days of the dataset
    '1yr': derive metrics from the 30 day period starting 1-year
           prior to the end of the dataset. this alings with the
           period from which our labels are derived.
  '''
  
  # determine how to group transaction data for metrics calculations
  grouping_fields = ['household_key']
  grouping_suffix = ''
  if include_commodity: 
    grouping_fields += ['commodity_desc']
    grouping_suffix = '_cmd'
    
  # get list of distinct grouping items in the original dataframe
  anchor_df = transactions.select(grouping_fields).distinct()
  
  # identify when dataset starts and ends
  min_day, max_day = (
    df
      .groupBy()
        .agg(
          f.min('day').alias('min_day'), 
          f.max('day').alias('max_day')
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
      .filter(f.expr(f"day between '{min_day}' and '{max_day}'")) # constrain to window
      .groupBy(grouping_fields)
        .agg(
          
          # summary metrics
          f.countDistinct('day').alias('days'), 
          f.countDistinct('basket_id').alias('baskets'),
          f.count('product_id').alias('products'), 
          f.count('*').alias('line_items'),
          f.sum('amount_list').alias('amount_list'),
          f.sum('instore_discount').alias('instore_discount'),
          f.sum('campaign_coupon_discount').alias('campaign_coupon_discount'),
          f.sum('manuf_coupon_discount').alias('manuf_coupon_discount'),
          f.sum('total_coupon_discount').alias('total_coupon_discount'),
          f.sum('amount_paid').alias('amount_paid'),
          
          # unique days with activity
          f.countDistinct(f.expr('case when instore_discount >0 then day else null end')).alias('days_with_instore_discount'),
          f.countDistinct(f.expr('case when campaign_coupon_discount >0 then day else null end')).alias('days_with_campaign_coupon_discount'),
          f.countDistinct(f.expr('case when manuf_coupon_discount >0 then day else null end')).alias('days_with_manuf_coupon_discount'),
          f.countDistinct(f.expr('case when total_coupon_discount >0 then day else null end')).alias('days_with_total_coupon_discount'),
          
          # unique baskets with activity
          f.countDistinct(f.expr('case when instore_discount >0 then basket_id else null end')).alias('baskets_with_instore_discount'),
          f.countDistinct(f.expr('case when campaign_coupon_discount >0 then basket_id else null end')).alias('baskets_with_campaign_coupon_discount'),
          f.countDistinct(f.expr('case when manuf_coupon_discount >0 then basket_id else null end')).alias('baskets_with_manuf_coupon_discount'),
          f.countDistinct(f.expr('case when total_coupon_discount >0 then basket_id else null end')).alias('baskets_with_total_coupon_discount'),          
    
          # unique products with activity
          f.countDistinct(f.expr('case when instore_discount >0 then product_id else null end')).alias('products_with_instore_discount'),
          f.countDistinct(f.expr('case when campaign_coupon_discount >0 then product_id else null end')).alias('products_with_campaign_coupon_discount'),
          f.countDistinct(f.expr('case when manuf_coupon_discount >0 then product_id else null end')).alias('products_with_manuf_coupon_discount'),
          f.countDistinct(f.expr('case when total_coupon_discount >0 then product_id else null end')).alias('products_with_total_coupon_discount'),          
    
          # unique line items with activity
          f.sum(f.expr('case when instore_discount >0 then 1 else null end')).alias('line_items_with_instore_discount'),
          f.sum(f.expr('case when campaign_coupon_discount >0 then 1 else null end')).alias('line_items_with_campaign_coupon_discount'),
          f.sum(f.expr('case when manuf_coupon_discount >0 then 1 else null end')).alias('line_items_with_manuf_coupon_discount'),
          f.sum(f.expr('case when total_coupon_discount >0 then 1 else null end')).alias('line_items_with_total_coupon_discount')          
          )    
    
      # per-day ratios
      .withColumn(f'baskets_per_day', f.expr('baskets/days'))
      .withColumn(f'products_per_day{window_suffix}', f.expr('products/days'))
      .withColumn(f'line_items_per_day', f.expr('line_items/days'))
      .withColumn(f'amount_list_per_day', f.expr('amount_list/days'))
      .withColumn(f'instore_discount_per_day', f.expr('instore_discount/days'))
      .withColumn(f'campaign_coupon_discount_per_day', f.expr('campaign_coupon_discount/days'))
      .withColumn(f'manuf_coupon_discount_per_day', f.expr('manuf_coupon_discount/days'))
      .withColumn(f'total_coupon_discount_per_day', f.expr('total_coupon_discount/days'))
      .withColumn(f'amount_paid_per_day', f.expr('amount_paid/days'))
      .withColumn(f'days_with_instore_discount_per_days', f.expr('days_with_instore_discount/days'))
      .withColumn(f'days_with_campaign_coupon_discount_per_days', f.expr('days_with_campaign_coupon_discount/days'))
      .withColumn(f'days_with_manuf_coupon_discount_per_days', f.expr('days_with_manuf_coupon_discount/days'))
      .withColumn(f'days_with_total_coupon_discount_per_days', f.expr('days_with_total_coupon_discount/days'))
    
      # per-day-in-set ratios
      .withColumn(f'days_to_days_in_set', f.expr(f'days/{days_in_window}'))
      .withColumn(f'baskets_per_days_in_set', f.expr(f'baskets/{days_in_window}'))
      .withColumn(f'products_to_days_in_set', f.expr(f'products/{days_in_window}'))
      .withColumn(f'line_items_per_days_in_set', f.expr(f'line_items/{days_in_window}'))
      .withColumn(f'amount_list_per_days_in_set', f.expr(f'amount_list/{days_in_window}'))
      .withColumn(f'instore_discount_per_days_in_set', f.expr(f'instore_discount/{days_in_window}'))
      .withColumn(f'campaign_coupon_discount_per_days_in_set', f.expr(f'campaign_coupon_discount/{days_in_window}'))
      .withColumn(f'manuf_coupon_discount_per_days_in_set', f.expr(f'manuf_coupon_discount/{days_in_window}'))
      .withColumn(f'total_coupon_discount_per_days_in_set', f.expr(f'total_coupon_discount/{days_in_window}'))
      .withColumn(f'amount_paid_per_days_in_set', f.expr(f'amount_paid/{days_in_window}'))
      .withColumn(f'days_with_instore_discount_per_days_in_set', f.expr(f'days_with_instore_discount/{days_in_window}'))
      .withColumn(f'days_with_campaign_coupon_discount_per_days_in_set', f.expr(f'days_with_campaign_coupon_discount/{days_in_window}'))
      .withColumn(f'days_with_manuf_coupon_discount_per_days_in_set', f.expr(f'days_with_manuf_coupon_discount/{days_in_window}'))
      .withColumn(f'days_with_total_coupon_discount_per_days_in_set', f.expr(f'days_with_total_coupon_discount/{days_in_window}'))

      # per-basket ratios
      .withColumn('products_per_basket', f.expr('products/baskets'))
      .withColumn('line_items_per_basket', f.expr('line_items/baskets'))
      .withColumn('amount_list_per_basket', f.expr('amount_list/baskets'))      
      .withColumn('instore_discount_per_basket', f.expr('instore_discount/baskets'))  
      .withColumn('campaign_coupon_discount_per_basket', f.expr('campaign_coupon_discount/baskets')) 
      .withColumn('manuf_coupon_discount_per_basket', f.expr('manuf_coupon_discount/baskets'))
      .withColumn('total_coupon_discount_per_basket', f.expr('total_coupon_discount/baskets'))    
      .withColumn('amount_paid_per_basket', f.expr('amount_paid/baskets'))
      .withColumn('baskets_with_instore_discount_per_baskets', f.expr('baskets_with_instore_discount/baskets'))
      .withColumn('baskets_with_campaign_coupon_discount_per_baskets', f.expr('baskets_with_campaign_coupon_discount/baskets'))
      .withColumn('baskets_with_manuf_coupon_discount_per_baskets', f.expr('baskets_with_manuf_coupon_discount/baskets'))
      .withColumn('baskets_with_total_coupon_discount_per_baskets', f.expr('baskets_with_total_coupon_discount/baskets'))
      
      # per-product ratios
      .withColumn('line_items_per_product', f.expr('line_items/products'))
      .withColumn('amount_list_per_product', f.expr('amount_list/products'))      
      .withColumn('instore_discount_per_product', f.expr('instore_discount/products'))  
      .withColumn('campaign_coupon_discount_per_product', f.expr('campaign_coupon_discount/products')) 
      .withColumn('manuf_coupon_discount_per_product', f.expr('manuf_coupon_discount/products'))
      .withColumn('total_coupon_discount_per_product', f.expr('total_coupon_discount/products'))    
      .withColumn('amount_paid_per_product', f.expr('amount_paid/products'))
      .withColumn('products_with_instore_discount_per_product', f.expr('products_with_instore_discount/products'))
      .withColumn('products_with_campaign_coupon_discount_per_product', f.expr('products_with_campaign_coupon_discount/products'))
      .withColumn('products_with_manuf_coupon_discount_per_product', f.expr('products_with_manuf_coupon_discount/products'))
      .withColumn('products_with_total_coupon_discount_per_product', f.expr('products_with_total_coupon_discount/products'))
      
      # per-line_item ratios
      .withColumn('amount_list_per_line_item', f.expr('amount_list/line_items'))      
      .withColumn('instore_discount_per_line_item', f.expr('instore_discount/line_items'))  
      .withColumn('campaign_coupon_discount_per_line_item', f.expr('campaign_coupon_discount/line_items')) 
      .withColumn('manuf_coupon_discount_per_line_item', f.expr('manuf_coupon_discount/line_items'))
      .withColumn('total_coupon_discount_per_line_item', f.expr('total_coupon_discount/line_items'))    
      .withColumn('amount_paid_per_line_item', f.expr('amount_paid/line_items'))
      .withColumn('products_with_instore_discount_per_line_item', f.expr('products_with_instore_discount/line_items'))
      .withColumn('products_with_campaign_coupon_discount_per_line_item', f.expr('products_with_campaign_coupon_discount/line_items'))
      .withColumn('products_with_manuf_coupon_discount_per_line_item', f.expr('products_with_manuf_coupon_discount/line_items'))
      .withColumn('products_with_total_coupon_discount_per_line_item', f.expr('products_with_total_coupon_discount/line_items'))    
    
      # amount_list ratios
      .withColumn('campaign_coupon_discount_to_amount_list', f.expr('campaign_coupon_discount/amount_list'))
      .withColumn('manuf_coupon_discount_to_amount_list', f.expr('manuf_coupon_discount/amount_list'))
      .withColumn('total_coupon_discount_to_amount_list', f.expr('total_coupon_discount/amount_list'))
      .withColumn('amount_paid_to_amount_list', f.expr('amount_paid/amount_list'))
      )
 

  # derive days-since metrics
  dayssince_df = (
    df
      .filter(f.expr(f"day <= '{max_day}'"))
      .groupBy(grouping_fields)
        .agg(
          f.min(f.expr(f"'{max_day}' - case when instore_discount >0 then day else '{min_day}' end")).alias('days_since_instore_discount'),
          f.min(f.expr(f"'{max_day}' - case when campaign_coupon_discount >0 then day else '{min_day}' end")).alias('days_since_campaign_coupon_discount'),
          f.min(f.expr(f"'{max_day}' - case when manuf_coupon_discount >0 then day else '{min_day}' end")).alias('days_since_manuf_coupon_discount'),
          f.min(f.expr(f"'{max_day}' - case when total_coupon_discount >0 then day else '{min_day}' end")).alias('days_since_total_coupon_discount')
          )
      )
  
  # combine metrics with anchor set to form return set 
  ret_df = (
    anchor_df
      .join(summary_df, on=grouping_fields, how='leftouter')
      .join(dayssince_df, on=grouping_fields, how='leftouter')
    )
  
  # rename fields based on control parameters
  for c in ret_df.columns:
    if c not in grouping_fields: # don't rename grouping fields
      ret_df = ret_df.withColumn(c, f.col(c).cast(DoubleType())) # cast all metrics as doubles to avoid confusion as categoricals
      ret_df = ret_df.withColumnRenamed(c,f'{c}{grouping_suffix}{window_suffix}')

  return ret_df

# COMMAND ----------

# MAGIC %md To derive features, we will expect to receive a dataframe of transactional data for a given period. From that set, we will need to identify the last day in the set as this will help us identify the point in time with which these features are associated.  We will then need to derive our features (using the function identified above) and then persist our data to a feature store table. To help us with this, we may define another function as follows:
# MAGIC 
# MAGIC **NOTE** The following code could be broken up into more reuseable units of logic but we've elected to collapse it into a single function to make the logic more transparent.

# COMMAND ----------

def generate_featureset(transactions_df, household_commodity_df):
  
  # IDENTIFY LAST DAY IN INCOMING DATAFRAME
  # --------------------------------------------
  last_day = (
    transactions_df
      .groupBy()
        .agg(f.max('day').alias('last_day')) # get last day in set
      .collect()
    )[0]['last_day']
  # --------------------------------------------
  
  # GET HOUSEHOLD FEATURES
  # --------------------------------------------
  # features will be grouped on households
  grouping_fields = ['household_key']

  # get master set of grouping field combinations
  features = household_commodity_df.select(grouping_fields).distinct()

  # get features and combine with other feature sets to form full feature set
  for window in ['30d','60d','90d','1yr']:
    features = (
      features
        .join(
            get_features(transactions_df, include_commodity=False, window=window), 
            on=grouping_fields, 
            how='leftouter'
          )
      )

  # fill-in any missing values
  household_features = features.fillna(value=0.0, subset=[c for c in features.columns if c not in grouping_fields])
  # --------------------------------------------
  
  # GET HOUSEHOLD-COMMODITY FEATURES
  # --------------------------------------------
  grouping_fields = ['household_key','commodity_desc']
  
  # get master set of grouping field combinations
  features = household_commodity_df.select(grouping_fields).distinct()

  # get period metrics
  for window in ['30d','60d','90d','1yr']:
    features = (
      features
        .join(
            get_features(transactions_df, include_commodity=True, window=window), 
            on=grouping_fields, 
            how='leftouter'
          )
      )

  # fill-in any missing values
  household_commodity_features = features.fillna(value=0.0, subset=[c for c in features.columns if c not in grouping_fields])
  # --------------------------------------------
  
  # COMBINE HOUSEHOLD & HOUSEHOLD-COMMODITY FEATURES
  # --------------------------------------------
  feature_set = (
    household_features
      .join(
        household_commodity_features,
        on='household_key'
        )
      .withColumn('day', f.lit(last_day)) # add last day to each record to identify day associated with features
      )
  # --------------------------------------------
  
  # WRITE DATA TO FEATURE STORE
  # --------------------------------------------
  # instantiate feature store client
  fs = FeatureStoreClient()
  
  # create feature store table (we will receive a warning with each call after the table has been created)
  _ = (
    fs
      .create_table(
        name='{0}.propensity_features'.format(config['database']), # name of feature store table
        primary_keys=['household_key','commodity_desc','day'], # name of keys that will be used to locate records
        schema=feature_set.schema, # schema of feature set as derived from our feature_set dataframe
        partition_columns=['day'], # partition data on a per-day basis
        description='features used for product propensity scoring' 
      )
    )
  
  # merge feature set data into feature store
  _ = (
    fs
      .write_table(
        name='{0}.propensity_features'.format(config['database']),
        df = feature_set,
        mode = 'merge' # merge data into existing feature set, instead of 'overwrite'
      )
    )
  # --------------------------------------------

# COMMAND ----------

# MAGIC %md ##Step 2: Generate Feature Sets
# MAGIC 
# MAGIC As described earlier, we will generate features for each of the last 30 days of our dataset. For each day in that range, we will grab the transactional data through the end of that day and pass that to the feature generation function defined above.  With each call, a new set of features will be persisted to the feature store for our later use:
# MAGIC 
# MAGIC **NOTE** As we only need the first and the last of these feature sets, we've added logic to skip over the days in-between.

# COMMAND ----------

# DBTITLE 1,Retrieve Transactional Data
transactions = (
    spark
      .table('transactions_adj')
      .join(
        spark.table('products'), # join to products to get commodity assignment
        on='product_id',
        how='inner'
        )
    )

# COMMAND ----------

# DBTITLE 1,Identify Last Day in Set
last_day = (
  transactions
    .groupBy()
      .agg(f.max('day').alias('last_day')) # get last day in set
    .collect()
  )[0]['last_day']

print(f"Last day in transaction set is {last_day.strftime('%Y-%m-%d')}")

# COMMAND ----------

# DBTITLE 1,Get Full Set of Households & Commodity Combinations
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
household_commodity_df = households.crossJoin(commodities)

# COMMAND ----------

# DBTITLE 1,Generate Features for Each of Last 30 Days in Set
# move last day 30 back from true last day
loop_last_day = last_day - timedelta(days=30)

# for each day until the true last day:
for d in range(31):
  
  # calculate "today"
  today = loop_last_day + timedelta(days=d)
  today = today.strftime('%Y-%m-%d')
  print(f"Calculating features for {today}")
  
  # generate features as of today
  if d in [0,30]: # skip any days not needed for remainder of demo
    generate_featureset( transactions.filter(f.expr(f"day <= '{today}'")), household_commodity_df )

# COMMAND ----------

# MAGIC %md Our features have been written to a feature store table named *propensity_features* within the *propensity* database (whichwe have set as our current database).  We can access this table as a standard SQL table to verify its contents as follows:

# COMMAND ----------

# DBTITLE 1,Verify Feature Record Counts by Day
display(
  spark
    .table('propensity_features')
    .groupBy('day')
      .agg(
        f.count('*').alias('records'),
        f.countDistinct('household_key').alias('households'),
        f.countDistinct('commodity_desc').alias('commodities')
        )
    .orderBy('day')
  )

# COMMAND ----------

# DBTITLE 1,Review Feature Records
display(
  spark
    .table('propensity_features')
    .orderBy('commodity_desc','day','household_key')
  )

# COMMAND ----------

# MAGIC %md While our feature store table can be accessed as a standard table, the real power of the feature store is in the metadata captured as the features are persisted to the structure.  You can see some of this metadata by accessing the table through the [FeatureStore UI](https://docs.databricks.com/applications/machine-learning/feature-store/ui.html):
# MAGIC 
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/propensity_feature%20store%20tables.png'>

# COMMAND ----------

# MAGIC %md By clicking on the table, you can see details about the table and its contents that may assist with its re-use.  But the real benefit of the feature store becomes visible as we train models against these structures and then later perform batch scoring:</p>
# MAGIC 
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/propensity_feature_store_table4.png'>

# COMMAND ----------

# MAGIC %md To delete any feature store table, use the Feature Store UI to delete the table. Once that is done, delete the underlying Delta Lake table as follows:
# MAGIC 
# MAGIC **NOTE** The code block is disabled by default to avoid accidental table deletion.

# COMMAND ----------

#_ = spark.catalog.setCurrentDatabase(config['database'])
#_ = spark.sql('DROP TABLE IF EXISTS propensity_features')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the [Databricks License](https://databricks.com/db-license-source).  All included or referenced third party libraries are subject to the licenses set forth below.

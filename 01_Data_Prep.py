# Databricks notebook source
# MAGIC %md The purpose of this notebook is to prepare the data required for the propensity scoring solution accelerator. This notebook is available at https://github.com/databricks-industry-solutions/propensity-workflows.

# COMMAND ----------

# MAGIC %md ## Introduction
# MAGIC
# MAGIC
# MAGIC For this solution accelerator, we envision a scenario where our data engineers are loading data in our lakehouse on an ongoing basis.  The marketing team periodically uses these data to estimate customer propensities for different product categories (*aka* commodities in the dataset we will use). Those propensities will be used by the marketing team to determine which advertisements, emailed product offers, *etc.* will be presented to individual customers.
# MAGIC
# MAGIC The dataset we will be using to support this scenario is the [*The Complete Journey*](https://www.kaggle.com/frtgnn/dunnhumby-the-complete-journey) dataset, published to Kaggle by Dunnhumby. The dataset consists of numerous files identifying household purchasing activity in combination with various promotional campaigns for about 2,500 households over a nearly 2 year period. The schema of the overall dataset may be represented as follows:
# MAGIC </p>
# MAGIC
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/segmentation_journey_schema3.png' width=500>
# MAGIC
# MAGIC The purpose of this notebook is to load this dataset into a Databricks database.

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
import pyspark.sql.functions as fn

# COMMAND ----------

# DBTITLE 1,Get Config Settings
# MAGIC %run ./util/config

# COMMAND ----------

# DBTITLE 1,Reinitiate the environment we use for this accelerator 
# This is defined in the `./util/config` notebook. It drops the database and feature store tables used in this accelerator
teardown()

# COMMAND ----------

# MAGIC %md ##Step 1: Setup Tables
# MAGIC
# MAGIC To make the data available for our analysis, we provide the `./util/extract_data` notebook to [download](https://www.kaggle.com/frtgnn/dunnhumby-the-complete-journey), extract and copy the data. Each file in the dataset is a comma-separated values file with a header which can be read to a table as follows:
# MAGIC
# MAGIC **NOTE** The paths used in this accelerator can be altered from within the `./util/config` notebook. 

# COMMAND ----------

# DBTITLE 1,Download and extract data
# MAGIC %run ./util/extract_data 

# COMMAND ----------

# DBTITLE 1,Define Function to Create Tables
def create_table(database_name, table_name, dbfs_file_path):
    
  # drop table if exists
  _ = spark.sql( f"DROP TABLE IF EXISTS {table_name}" )

  # read data from input file
  df = ( 
    spark
     .read
       .csv(
         dbfs_file_path,
         header=True,
         inferSchema=True
         )
    )
  
  # convert day integers to actual dates
  # for each column
  for c in df.columns:
    # if column is an integer day column
    if c.lower().endswith('day'):
      # convert it to a date value
      df = df.withColumn(c, fn.expr(f"date_add('2018-01-01', cast({c} as int)-1)"))
  
  # write data to table
  _ = (
    df
     .write
       .format('delta')
       .mode('overwrite')
       .option('overwriteSchema', 'true')
       .saveAsTable(table_name)
     )

# COMMAND ----------

# MAGIC %md It's important to note the dates used in this data set are not proper dates.  Instead, they are integer values ranging from 1 to 711 where 1 represents the first date in the range and 711 indicates the last.  To make it easier to explore how time-oriented values would be employed in later steps, our function converts these to actual dates by making day 1 equal to January 1, 2018 and adjusting the other values relative to this.  The use of this specific date is arbitrary and doesn't reflect any known dates associated with the original dataset.
# MAGIC
# MAGIC  From there, we might convert the CSV files to accessible tables as follows:

# COMMAND ----------

# DBTITLE 1,Create Tables
create_table( config['database_name'], 'transactions',  f"{config['mount_point']}/bronze/transaction_data.csv")
create_table( config['database_name'], 'products',  f"{config['mount_point']}/bronze/product.csv")
create_table( config['database_name'], 'households',  f"{config['mount_point']}/bronze/hh_demographic.csv")
create_table( config['database_name'], 'coupons',  f"{config['mount_point']}/bronze/coupon.csv")
create_table( config['database_name'], 'campaigns',  f"{config['mount_point']}/bronze/campaign_desc.csv")
create_table( config['database_name'], 'coupon_redemptions',  f"{config['mount_point']}/bronze/coupon_redempt.csv")
create_table( config['database_name'], 'campaigns_households',  f"{config['mount_point']}/bronze/campaign_table.csv")
create_table( config['database_name'], 'causal_data',  f"{config['mount_point']}/bronze/causal_data.csv")

# COMMAND ----------

# DBTITLE 1,Verify Tables Exist
# MAGIC %sql SHOW TABLES;

# COMMAND ----------

# MAGIC %md ##Step 2: Adjust Transactional Data
# MAGIC
# MAGIC The transactional data will be the focal point of our analysis.  It contains information about what was purchased by each household and when along with various discounts applied at the time of purchase. Some of this information is presented in a manner that is not easily consumable.  As such, we will implement some simple logic to sum discounts and combine these with amounts paid to recreate list pricing and make other simple adjustments that make the transactional data a bit easier to consume:  

# COMMAND ----------

# DBTITLE 1,Review Structure of Transactions Table
# MAGIC %sql  DESCRIBE transactions;

# COMMAND ----------

# DBTITLE 1,Create Adjusted Transactions Table
# MAGIC %sql
# MAGIC
# MAGIC CREATE OR REPLACE TABLE transactions_adj
# MAGIC USING DELTA
# MAGIC AS
# MAGIC   SELECT
# MAGIC     household_key,
# MAGIC     basket_id,
# MAGIC     week_no,
# MAGIC     day,
# MAGIC     trans_time,
# MAGIC     store_id,
# MAGIC     product_id,
# MAGIC     amount_list,
# MAGIC     campaign_coupon_discount,
# MAGIC     manuf_coupon_discount,
# MAGIC     manuf_coupon_match_discount,
# MAGIC     total_coupon_discount,
# MAGIC     instore_discount,
# MAGIC     amount_paid,
# MAGIC     units
# MAGIC   FROM (
# MAGIC     SELECT 
# MAGIC       household_key,
# MAGIC       basket_id,
# MAGIC       week_no,
# MAGIC       day,
# MAGIC       trans_time,
# MAGIC       store_id,
# MAGIC       product_id,
# MAGIC       COALESCE(sales_value - retail_disc - coupon_disc - coupon_match_disc,0.0) as amount_list,
# MAGIC       CASE 
# MAGIC         WHEN COALESCE(coupon_match_disc,0.0) = 0.0 THEN -1 * COALESCE(coupon_disc,0.0) 
# MAGIC         ELSE 0.0 
# MAGIC         END as campaign_coupon_discount,
# MAGIC       CASE 
# MAGIC         WHEN COALESCE(coupon_match_disc,0.0) != 0.0 THEN -1 * COALESCE(coupon_disc,0.0) 
# MAGIC         ELSE 0.0 
# MAGIC         END as manuf_coupon_discount,
# MAGIC       -1 * COALESCE(coupon_match_disc,0.0) as manuf_coupon_match_discount,
# MAGIC       -1 * COALESCE(coupon_disc - coupon_match_disc,0.0) as total_coupon_discount,
# MAGIC       COALESCE(-1 * retail_disc,0.0) as instore_discount,
# MAGIC       COALESCE(sales_value,0.0) as amount_paid,
# MAGIC       quantity as units
# MAGIC     FROM transactions
# MAGIC     );

# COMMAND ----------

# DBTITLE 1,Review Adjusted Transactions Data
# MAGIC %sql  SELECT * FROM transactions_adj LIMIT 100;

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the [Databricks License](https://databricks.com/db-license-source).  All included or referenced third party libraries are subject to the licenses set forth below.

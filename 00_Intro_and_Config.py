# Databricks notebook source
# MAGIC %md The purpose of this notebook is to set the various configuration values that will control both interactive and workflow components of the propensity scoring solution accelerator.  This notebook is available at https://github.com/databricks-industry-solutions/propensity-workflows.

# COMMAND ----------

# MAGIC %md ##Introduction
# MAGIC
# MAGIC The purpose of this solution accelerator is to setup a high-level workflow with which household (customer) propensity scores for various product categories can be updated on a regular basis. Such scores could be used by a variety of marketing systems to identify classes of products or product-aligned promotional offers to present to users as they interact with a website, mobile app or email messages sent to them.
# MAGIC
# MAGIC We envision this workflow as consisting of three principal tasks:
# MAGIC </p>
# MAGIC
# MAGIC 1. Feature Generation (addressed in notebook *04a_Task__Feature_Engineering*)
# MAGIC 2. Model Training  (addressed in notebook *04b_TASK__Model_Training*)
# MAGIC 3. Propensity Scoring (addressed in notebook *04c_TASK__Propensity_Estimation*)
# MAGIC
# MAGIC
# MAGIC **NOTE** Our focus in the numbered notebooks will be on workflow enablement. To understand the details of the work involved in each of these steps, please review the content in each of the task-aligned notebooks.
# MAGIC
# MAGIC At the time of solution initialization, the team responsible for the solution will need to create the backlog of features required to support model training and then complete a first pass on the model training itself.  At that point, propensity scores can be created through a two-part workflow, one of which operates daily and the other of which operates less frequently, most likely weekly.
# MAGIC
# MAGIC In the daily workflow, features are calculated from the latest information available in the system.  Those features are used in combination with available models to assemble the set of propensity scores to be used by marketers.
# MAGIC
# MAGIC In the weekly workflow, models are retrained using pre-calculated features.  The models are moved into production-ready status so that the next iteration of the daily workflow can leverage them for their work.
# MAGIC <p>
# MAGIC
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/prop_workflow.png' width=800>
# MAGIC <p>
# MAGIC
# MAGIC In the notebooks numbered 1 & 2, we will tackle the initialization steps, calling tasks associated with the three stages identified above in a slightly different sequence in order to setup our environment.  In notebook 3, we will setup the daily and weekly workflows as described above.

# COMMAND ----------

# MAGIC %md ##Step 1: Establish Configuration Settings

# COMMAND ----------

# MAGIC %run ./util/config

# COMMAND ----------

# MAGIC %md ##Step 2: Make Configuration Settings Accessible in Workflows
# MAGIC
# MAGIC The configuration settings established here are needed by the various tasks that make up our production workflows.  Unlike interactive notebooks which can run another notebook and directly access any variables defined within them, tasks in a Databricks Workflow execute in a more standalone manner.  That said, we can pass values between tasks in a workflow by using the Databricks Utility [Jobs utility](https://docs.databricks.com/dev-tools/databricks-utils.html#jobs-utility-dbutilsjobs) object as shown here:
# MAGIC
# MAGIC **NOTE** The method calls in the next cell will fail if this code is run outside a Databricks workflow. The `try...except...` block is intended to catch that so that this notebook can be run in both interactive and workflow modes.

# COMMAND ----------

# DBTITLE 1,Make Accessible Needed Config Values
try:
  dbutils.jobs.taskValues.set('database_name', config['database_name'])
  dbutils.jobs.taskValues.set('current_day', config['current_day'])
except:
  pass

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the [Databricks License](https://databricks.com/db-license-source).  All included or referenced third party libraries are subject to the licenses set forth below.

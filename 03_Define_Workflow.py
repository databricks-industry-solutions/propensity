# Databricks notebook source
# MAGIC %md The purpose of this notebook is to setup the workflows required for the propensity scoring solution accelerator. This notebook is available at https://github.com/databricks-industry-solutions/propensity-workflows.

# COMMAND ----------

# MAGIC %md ##Introduction
# MAGIC
# MAGIC In this notebook, we assemble the daily and weekly workflows through which customer propensities will be kept up to date:
# MAGIC </p>
# MAGIC
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/prop_workflow.png' width=800>
# MAGIC
# MAGIC The daily workflow will focus on executing the *04a_Task__Feature_Engineering* and *04c_Task__Propensity_Estimation* tasks. The weekly workflow will focus on executing the *04b_Task__Model_Training* task. More details about the logic associated with each task can be found within the individual task notebooks.  
# MAGIC
# MAGIC In addition to the UI instructions listed below, we provide automation in the `./RUNME` notebook that can create these workflows for you. 
# MAGIC

# COMMAND ----------

# DBTITLE 1,Get Config Settings
# MAGIC %run ./util/config

# COMMAND ----------

# MAGIC %md ##Step 1: Define the Daily Workflow
# MAGIC
# MAGIC Using the left-hand navigation bar in the Databricks workspace, select ***Workflows***.  You'll need to be in either the *Data Science & Engineering* or the *Machine Learning* configuration to see this item.
# MAGIC </p>
# MAGIC
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/propensity_workflows_button.PNG' width=200>
# MAGIC
# MAGIC Within Workflows, click the ***Create Job*** button. 
# MAGIC

# COMMAND ----------

# MAGIC %md For the first task, assign it the following values:
# MAGIC </p>
# MAGIC
# MAGIC * **Task name:** Get_Config
# MAGIC * **Type:** Notebook
# MAGIC * **Source:** Workspace
# MAGIC * **Path:** *navigate to the path of the 00_Intro_and_Config notebook*
# MAGIC
# MAGIC For cluster, you may want to create a new job cluster.  To do this, click the pencil icon on the default entry in the *Cluster* drop-down.  Give it a name of *PropensityScoring-Daily* and configuring it the same as the cluster on which you ran your notebooks 1 & 2.  (You may want to add more workers depending on if you want to speed up some of the steps.) The remainder of the options can be configured based on your preferences.
# MAGIC
# MAGIC </p>
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/propensity_get_config_task.PNG' width=1000>
# MAGIC
# MAGIC Click the ***Create*** button at the bottom of the screen.  When you do this, you will be asked to assign this job a name.  Use the name *PropensityScoring-Daily*. Click *Confirm and create* to continue.

# COMMAND ----------

# MAGIC %md With the first task created, you can now click the **plus sign** at the center of the screen to add the next task.  As you click the plus sign, select *Notebook* to indicate the task will be a notebook type.  
# MAGIC
# MAGIC </p>
# MAGIC
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/propensity_add_task.PNG' width=400>
# MAGIC
# MAGIC Assign this new task the following values:
# MAGIC </p>
# MAGIC
# MAGIC * **Task name:** Generate_Features
# MAGIC * **Type:** Notebook
# MAGIC * **Source:** Workspace
# MAGIC * **Path:** *navigate to the path of the 04a_Task__Feature_Engineering notebook*
# MAGIC * **Cluster:** PropensityScoring-Daily
# MAGIC
# MAGIC Click the ***Create task*** button.

# COMMAND ----------

# MAGIC %md Click the **plus sign** again to add the next notebook task.  Assign this task the following values:
# MAGIC </p>
# MAGIC
# MAGIC * **Task name:** Estimate_Propensity
# MAGIC * **Type:** Notebook
# MAGIC * **Source:** Workspace
# MAGIC * **Path:** *navigate to the path of the 04c_Task__Propensity_Estimation notebook*
# MAGIC * **Cluster:** PropensityScoring-Daily
# MAGIC
# MAGIC Click ***Create task*** to complete the process.
# MAGIC </p>
# MAGIC
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/propensity_daily_workflow.PNG' width=1000>

# COMMAND ----------

# MAGIC %md You've now defined your daily workflow.  You can click the ***Run now*** button in the upper right-hand corner of the screen to run the workflow end to end and verify it works as expected.  To monitor this run, click on the *Runs* tab in the upper left-hand corner of the job definition screen. A successful run should look like the following:
# MAGIC </p>
# MAGIC
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/propensity_daily_success.PNG' width=1000>
# MAGIC
# MAGIC If you encounter an error, simply click on the task that failed to examine the notebook and error that occurred. Correct the error in the original notebooks and re-start the failed task to complete the workflow.
# MAGIC
# MAGIC Once that run successfully completes, you can then use the navigation in the right-hand pane to put the job on a schedule.  Here we have scheduled our job to run at 3:30 AM every morning:
# MAGIC </p>
# MAGIC
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/propensity_daily_scheduled.PNG' width=800>

# COMMAND ----------

# MAGIC %md ##Step 2: Define the Weekly Workflow
# MAGIC
# MAGIC Next, we define a task for the weekly workflow.  The steps are very similar to the ones outlined above except the sequence of steps need to call the following notebooks:
# MAGIC </p>
# MAGIC
# MAGIC 1. *00_Intro_and_Config*
# MAGIC 2. *04b_Task__Model_Training*
# MAGIC
# MAGIC Once the job has been manually run to confirm its in good working order, you can set it on a schedule to run once a week.  We have scheduled our job to run at 2 AM on Sunday morning.

# COMMAND ----------

# MAGIC %md ##Step 3: Monitor Workflow Progress
# MAGIC
# MAGIC With the jobs on schedule, we can check in periodically to verify that things have run as expected.  To review the dashboard for a given job, simply return to the ***Workflows*** interface, click on the job, and make sure you are on the *Runs* tab.  
# MAGIC </p>
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/propensity_daily_workflow_monitor.PNG' width=1000>
# MAGIC </p>
# MAGIC
# MAGIC Should you encounter an error, the task that failed will be highlighted in red:
# MAGIC </p>
# MAGIC
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/propensity_weekly_failed.PNG' width=1000>
# MAGIC
# MAGIC </p>
# MAGIC Simply click on the failing task, highlight the error, and return to the original notebook to make the appropriate changes.  Once those changes are in place, you can restart the task by returning to the failed workflow instance and clicking the **Repair run** button (in the upper right-hand corner of the screen).  This will restart the failed task (and any downstream tasks) which providing it access to job state information.
# MAGIC
# MAGIC One last thing to point out, if you don't want to regularly return to the Workflows table to check on the status of your jobs, you can enable email and Slack-based (as well as other forms of ) notifications using the [workflows notifications](https://docs.databricks.com/workflows/jobs/job-notifications.html) capability.  You can find more details about the supported notifications destinations [here](https://docs.databricks.com/sql/admin/notification-destinations.html).

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the [Databricks License](https://databricks.com/db-license-source).  All included or referenced third party libraries are subject to the licenses set forth below.

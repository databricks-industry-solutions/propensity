# Databricks notebook source
# MAGIC %md This notebook sets up the companion cluster(s) to run the solution accelerator. It also creates the Workflow to illustrate the order of execution. Happy exploring! 
# MAGIC 🎉
# MAGIC
# MAGIC **Steps**
# MAGIC 1. Simply attach this notebook to a cluster and hit Run-All for this notebook. A multi-step job and the clusters used in the job will be created for you and hyperlinks are printed on the last block of the notebook. 
# MAGIC
# MAGIC 2. Run the accelerator notebooks: Feel free to explore the multi-step job page and **run the Workflow**, or **run the notebooks interactively** with the cluster to see how this solution accelerator executes. 
# MAGIC
# MAGIC     2a. **Run the Workflow**: Navigate to the Workflow link and hit the `Run Now` 💥. 
# MAGIC   
# MAGIC     2b. **Run the notebooks interactively**: Attach the notebook with the cluster(s) created and execute as described in the `job_json['tasks']` below.
# MAGIC
# MAGIC **Prerequisites** 
# MAGIC 1. You need to have cluster creation permissions in this workspace.
# MAGIC
# MAGIC 2. In case the environment has cluster-policies that interfere with automated deployment, you may need to manually create the cluster in accordance with the workspace cluster policy. The `job_json` definition below still provides valuable information about the configuration these series of notebooks should run with. 
# MAGIC
# MAGIC **Notes**
# MAGIC 1. The pipelines, workflows and clusters created in this script are not user-specific. Keep in mind that rerunning this script again after modification resets them for other users too.
# MAGIC
# MAGIC 2. If the job execution fails, please confirm that you have set up other environment dependencies as specified in the accelerator notebooks. Accelerators may require the user to set up additional cloud infra or secrets to manage credentials. 

# COMMAND ----------

# DBTITLE 0,Install util packages
# MAGIC %pip install git+https://github.com/databricks-academy/dbacademy@v1.0.13 git+https://github.com/databricks-industry-solutions/notebook-solution-companion@safe-print-html --quiet --disable-pip-version-check

# COMMAND ----------

from solacc.companion import NotebookSolutionCompanion

# COMMAND ----------

# MAGIC %md
# MAGIC Since this accelerator uses data from a Kaggle competition, we need to accept the competition [rules](https://www.kaggle.com/competitions/demand-forecasting-kernels-only/data) and set up a few credentials in order to access the Kaggle dataset. Grab the key for your Kaggle account ([documentation](https://www.kaggle.com/docs/api#getting-started-installation-&-authentication) here). Here we demonstrate using the [Databricks Secret Scope](https://docs.databricks.com/security/secrets/secret-scopes.html) for credential management. 
# MAGIC
# MAGIC Copy the block of code below, replace the name the secret scope and fill in the credentials and execute the block. After executing the code, The accelerator notebook will be able to access the credentials it needs.
# MAGIC
# MAGIC
# MAGIC ```
# MAGIC client = NotebookSolutionCompanion().client
# MAGIC try:
# MAGIC   client.execute_post_json(f"{client.endpoint}/api/2.0/secrets/scopes/create", {"scope": "solution-accelerator-cicd"})
# MAGIC except:
# MAGIC   pass
# MAGIC client.execute_post_json(f"{client.endpoint}/api/2.0/secrets/put", {
# MAGIC   "scope": "solution-accelerator-cicd",
# MAGIC   "key": "kaggle_username",
# MAGIC   "string_value": "____"
# MAGIC })
# MAGIC
# MAGIC client.execute_post_json(f"{client.endpoint}/api/2.0/secrets/put", {
# MAGIC   "scope": "solution-accelerator-cicd",
# MAGIC   "key": "kaggle_key",
# MAGIC   "string_value": "____"
# MAGIC })
# MAGIC ```

# COMMAND ----------

# MAGIC %md Here we define a workflow to run the main body of the accelerator which includes all the notebooks. 

# COMMAND ----------

job_json = {
        "timeout_seconds": 28800,
        "max_concurrent_runs": 1,
        "tags": {
            "usage": "solacc_testing",
            "group": "RCG"
        },
        "tasks": [
            {
                "job_cluster_key": "propensity_workflow_cluster",
                "notebook_task": {
                    "notebook_path": f"00_Intro_and_Config"
                },
                "task_key": "00_Intro_and_Config"
            },
            {
                "job_cluster_key": "propensity_workflow_cluster",
                "notebook_task": {
                    "notebook_path": f"01_Data_Prep"
                },
                "task_key": "01_Data_Prep",
                "depends_on": [
                    {
                        "task_key": "00_Intro_and_Config"
                    }
                ]
            },
            {
                "job_cluster_key": "propensity_workflow_cluster",
                "notebook_task": {
                    "notebook_path": f"02_Initialize_Solution"
                },
                "task_key": "02_Initialize_Solution",
                "depends_on": [
                    {
                        "task_key": "01_Data_Prep"
                    }
                ]
            },
            {
                "job_cluster_key": "propensity_workflow_cluster",
                "notebook_task": {
                    "notebook_path": f"03_Define_Workflow"
                },
                "task_key": "03_Define_Workflow",
                "depends_on": [
                    {
                        "task_key": "02_Initialize_Solution"
                    }
                ]
            },
            {
                "job_cluster_key": "propensity_workflow_cluster",
                "notebook_task": {
                    "notebook_path": f"04a_Task__Feature_Engineering"
                },
                "task_key": "04a_Task__Feature_Engineering",
                "depends_on": [
                    {
                        "task_key": "03_Define_Workflow"
                    }
                ]
            },
            {
                "job_cluster_key": "propensity_workflow_cluster",
                "notebook_task": {
                    "notebook_path": f"04b_Task__Model_Training"
                },
                "task_key": "04b_Task__Model_Training",
                "depends_on": [
                    {
                        "task_key": "04a_Task__Feature_Engineering"
                    }
                ]
            },
            {
                "job_cluster_key": "propensity_workflow_cluster",
                "notebook_task": {
                    "notebook_path": f"04c_Task__Propensity_Estimation"
                },
                "task_key": "04c_Task__Propensity_Estimation",
                "depends_on": [
                    {
                        "task_key": "04b_Task__Model_Training"
                    }
                ]
            }
        ],
        "job_clusters": [
            {
                "job_cluster_key": "propensity_workflow_cluster",
                "new_cluster": {
                    "spark_version": "12.2.x-cpu-ml-scala2.12",
                "spark_conf": {
                    "spark.databricks.delta.formatCheck.enabled": "false"
                    },
                    "num_workers": 4,
                    "node_type_id": {"AWS": "i3.xlarge", "MSA": "Standard_DS3_v2", "GCP": "n1-highmem-4"},
                    "custom_tags": {
                        "usage": "solacc_testing"
                    },
                }
            }
        ]
    }

# COMMAND ----------

dbutils.widgets.dropdown("run_job", "False", ["True", "False"])
run_job = dbutils.widgets.get("run_job") == "True"
NotebookSolutionCompanion().deploy_compute(job_json, run_job=run_job)

# COMMAND ----------

# MAGIC %md In notebook 03 we show how to define a daily workflow for scoring and a weekly workflow for model retraining. Here we include some automation to create the same workflows as in the UI screenshots.

# COMMAND ----------

job_json_daily = {
      "name": "[Solution Accelerator] Propensity Scoring - Daily",
      "timeout_seconds": 28800,
      "max_concurrent_runs": 1,
      "tasks": [
          {
              "task_key": "Get_Config",
              "notebook_task": {
                  "notebook_path": "00_Intro_and_Config"
              },
              "job_cluster_key": "propensity_daily_cluster"
          },
          {
              "task_key": "Generate_Features",
              "depends_on": [
                  {
                      "task_key": "Get_Config"
                  }
              ],
              "notebook_task": {
                  "notebook_path": "04a_Task__Feature_Engineering"
              },
              "job_cluster_key": "propensity_daily_cluster"
          },
          {
              "task_key": "Estimate_Propensity",
              "depends_on": [
                  {
                      "task_key": "Generate_Features"
                  }
              ],
              "notebook_task": {
                  "notebook_path": "04c_Task__Propensity_Estimation"
              },
              "job_cluster_key": "propensity_daily_cluster"
          }
      ],
      "job_clusters": [
          {
              "job_cluster_key": "propensity_daily_cluster",
              "new_cluster": {
                  "cluster_name": "",
                  "spark_version": "12.2.x-cpu-ml-scala2.12",
                  "spark_conf": {
                      "spark.databricks.delta.preview.enabled": "true"
                  },
                  "node_type_id": {"AWS": "i3.xlarge", "MSA": "Standard_DS3_v2", "GCP": "n1-highmem-4"},
                  "num_workers": 4
              }
          }
      ]
  }

job_json_weekly = {
      "name": "[Solution Accelerator] Propensity Workflow - Weekly",
      "timeout_seconds": 28800,
      "max_concurrent_runs": 1,
      "tasks": [
          {
              "task_key": "Config",
              "notebook_task": {
                  "notebook_path": "00_Intro_and_Config"
              },
              "job_cluster_key": "propensity_weekly_cluster"
          },
          {
              "task_key": "Train_Model",
              "depends_on": [
                  {
                      "task_key": "Config"
                  }
              ],
              "notebook_task": {
                  "notebook_path": "04b_Task__Model_Training"
              },
              "job_cluster_key": "propensity_weekly_cluster"
          }
      ],
      "job_clusters": [
          {
              "job_cluster_key": "propensity_weekly_cluster",
              "new_cluster": {
                  "cluster_name": "",
                  "spark_version": "12.2.x-cpu-ml-scala2.12",
                  "spark_conf": {
                      "spark.databricks.delta.preview.enabled": "true"
                  },
                  "node_type_id": {"AWS": "i3.xlarge", "MSA": "Standard_DS3_v2", "GCP": "n1-highmem-4"},
                  "num_workers": 4
              }
          }
      ]
  }

# COMMAND ----------

NotebookSolutionCompanion().deploy_compute(job_json_daily, run_job=run_job)
NotebookSolutionCompanion().deploy_compute(job_json_weekly, run_job=run_job)

# COMMAND ----------



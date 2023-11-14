from datetime import datetime
import os
from airflow import DAG
from btx.plugins.jid import JIDSlurmOperator

# DAG SETUP
description='BTX test DAG'
dag_name = os.path.splitext(os.path.basename(__file__))[0]

dag = DAG(
    dag_name,
    start_date=datetime( 2022,4,1 ),
    schedule_interval=None,
    description=description,
  )


# Tasks SETUP
task_id='visualize_sample'
visualize_sample = JIDSlurmOperator( task_id=task_id, dag=dag)

task_id='determine_cell'
determine_cell = JIDSlurmOperator( task_id=task_id, dag=dag)

# Draw the DAG
visualize_sample
determine_cell

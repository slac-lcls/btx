from datetime import datetime
import os
from airflow import DAG
from plugins.jid import JIDSlurmOperator

description='BTX process SAXS DAG'
dag_name = os.path.splitext(os.path.basename(__file__))[0]

dag = DAG(
    dag_name,
    start_date=datetime( 2022,4,1 ),
    schedule_interval=None,
    description=description,
  )

task_id='run_analysis'
run_analysis = JIDSlurmOperator(task_id=task_id, dag=dag)

task_id='plot_saxs'
plot_saxs = JIDSlurmOperator(task_id=task_id, dag=dag)

run_analysis >> plot_saxs

# Query Generation

Use the scripts in this repo to generate queries for a given context following the [doc2query](https://huggingface.co/doc2query/msmarco-t5-base-v1) method.


## Data Details

We have converted 1st shard of the pile dataset into 64 smaller sub-shards. Each shard contains upto 100,000 records. Each record contains text and metadata. Out goal is to generate multiple queries that the text could answer and append it to the original record. This should improve retrieval.

The data is stored in S3 laion bucket with the following path: `open-assistant-retrieval/doc2query_pile/stage_0/pile-<shard_index>.jsonl.gz`


## Execution

At this time, the script `generate_queries_main.py` runs the query generation process in parallel. Each process runs the query generation for one shard as a separate python process. We don't use python multiprocessing because after 4-6 parallel workers, there are not much benefits in running multiprocessing. But with the current approach, we can launch as many parallel workers as we like based on the number of cpus. The only downside is that for killing the job, we would need to use `pkill -9 python`.


Modify the parameters as required in the `generate_queries_main.py` such as number of workers.


Run the script as follows:
```
python3 generate_queries_main.py
```


## Understanding Output Files


Each worker executes one shard at a time. The worker process copies the shard from S3 to the local tmp directory (specified in the params). Three files are generated for each worker process in the following format:
1. `proc_<worker_id>`.log: Logs the std out of the worker 
2. `q-<index>-pile-<shard_index>.jsonl.gz`: For each record, the generated queries are added to the record and stored in this file. `shard_index` denotes the shard file and `index` is number that just indicates the order of the file. For example, if the job was killed for any reasons and 150 records we completed, the restart will create a new file `q-151-pile-000.jsonl.gz` as a continuation from `q-000-pile-000.jsonl.gz`. We would then just combine all these files into one file for a shard. This is done so we can stop and restart at anytime without losing progress.
3. `status-pile-<shard_index>.txt`: This file keeps tracks of the indices of records that have been completed for a given shard. So when we restart, we can just continue from the last index found in the status file.


## Exectuing Batch Job on Stability Cluster
I tried running the `launch_generation_job2.sh` script on the Stability cluster, it does not run. We still need to figure out the reason and make it work.
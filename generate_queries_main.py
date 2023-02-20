import sys
import os
import torch
from multiprocessing import Pool


def execute_query_generation(s3_file_path, batch_size, device_id, model_name, bucket_name,
                             tmp_dir, worker_id):
    cmd = (f"nohup python3 /fsx/home-lkc/doc2query-experiments/query_generation.py "
           f"--s3_file_path {s3_file_path} "
           f"--batch_size {batch_size} "
           f"--device_id {device_id} "
           f"--model_name {model_name} "
           f"--bucket_name {bucket_name} "
           f"--tmp_dir {tmp_dir} "
           f"--worker_id {worker_id} > /fsx/home-lkc/doc2query-experiments/tmp/proc_{worker_id}.log 2>&1 &")
    print(cmd, '\n')
    os.system(cmd)

    
def main():
    batch_size = 20
    # skip = [0, 1]
    skip = list(range(2, 64))
    s3_file_paths = [f"open-assistant-retrieval/doc2query_pile/stage_0/pile-{x:03d}.jsonl.gz" for x in range(64)
                     if x not in skip]
    # s3_file_paths = s3_file_paths[:8]
    model_name = 'doc2query/msmarco-t5-base-v1'
    bucket_name = 's-laion'
    tmp_dir = '/fsx/home-lkc/doc2query-experiments/tmp'
    num_devices = torch.cuda.device_count()
    num_workers = 2

    N = len(s3_file_paths)

    device_ids = [x % num_devices for x in range(N)]
    worker_ids = [x % num_workers for x in range(N)]

    #print(s3_file_paths)
    #sys.exit()
    

    with Pool(num_workers) as p:
        p.starmap(execute_query_generation, zip(s3_file_paths,
                                                [batch_size] * N,
                                                device_ids,
                                                [model_name] * N,
                                                [bucket_name] * N,
                                                [tmp_dir] * N,
                                                worker_ids))
        
            
    

if __name__ == '__main__':
    main()

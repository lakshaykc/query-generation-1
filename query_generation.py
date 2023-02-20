import os
import gzip
import torch
import json
import sys
import pdb
import argparse
from time import time
from pathlib import Path
from multiprocessing import Pool
from transformers import T5Tokenizer, T5ForConditionalGeneration


class QueryGeneration(object):
    def __init__(self, s3_file_path, batch_size, device_id, model_name, bucket_name,
                 tmp_dir, worker_id):
        self.bucket_name = bucket_name
        self.s3_file_path = s3_file_path
        self.batch_size = batch_size
        self.tmp_dir = tmp_dir
        self.device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.worker_id = worker_id
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model = self.model.half().to(self.device) if torch.cuda.is_available() else self.model.to(self.device)

        Path(self.tmp_dir).mkdir(parents=True, exist_ok=True)

        self.status_file_path = os.path.join(self.tmp_dir, 'status-' + os.path.basename(self.s3_file_path))
        self.status_file_path = self.status_file_path.replace('.jsonl.gz', '.txt')
        self.local_file_path = os.path.join(self.tmp_dir, os.path.basename(self.s3_file_path))
        self.start_index = self.get_start_idx()
        print(f"start_index: {self.start_index}. Lines preceding this index have been completed.")
        sys.stdout.flush()
        self.local_write_file_path = self.get_local_write_file_path()

    def get_local_write_file_path(self):
        local_write_file_path = os.path.join(self.tmp_dir,
                                             f'q-{self.start_index:06d}-' + os.path.basename(self.s3_file_path))
        return local_write_file_path

    def get_start_idx(self):
        if os.path.exists(self.status_file_path):
            with open(self.status_file_path, 'r') as f:
                lines = f.readlines()
                try:
                    start_idx = int(lines[-1].strip())
                except IndexError:
                    start_idx = 0
        else:
            start_idx = 0
        return start_idx

    def download_file_from_s3(self):
        if os.path.exists(self.local_file_path):
            print(f"File already exists locally: {self.local_file_path}")
            sys.stdout.flush()
            return
        # use aws cli to download file from s3
        os.system(f"aws s3 cp s3://{self.bucket_name}/{self.s3_file_path} {self.local_file_path}")
        assert os.path.exists(self.local_file_path)

    def tokenize(self, inputs, padding=True, truncation=True, add_special_tokens=True, max_length=290):
        inputs = self.tokenizer.batch_encode_plus(
            inputs,
            max_length=max_length,
            add_special_tokens=add_special_tokens,
            truncation=truncation,
            padding="max_length" if padding else False,
            pad_to_max_length=padding,
            return_tensors="pt"
        )
        return inputs

    def generate_questions(self, inputs):
        inputs = self.tokenize(inputs, padding=True, truncation=True)

        outs = self.model.generate(
            input_ids=inputs['input_ids'].to(self.device),
            attention_mask=inputs['attention_mask'].to(self.device),
            max_length=64,
            do_sample=True,
            top_p=0.95,
            num_return_sequences=20
        )

        questions = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
        return questions

    def process(self):
        self.download_file_from_s3()
        with gzip.open(self.local_file_path, 'rt') as read_file, \
                gzip.open(self.local_write_file_path, 'w') as write_file, \
                open(self.status_file_path, 'a') as status_file:

            t0 = time()
            for i, line in enumerate(read_file):
                if i < self.start_index:
                    continue
                count = i - self.start_index + 1

                if i % 20 == 0 and i != 0:
                    print(f"Worker {self.worker_id}: Processed {count} records in {time() - t0:.2f} seconds.")
                    sys.stdout.flush()
                    t0 = time()
                data = json.loads(line)
                try:
                    questions = self.generate_questions([data['text']])
                    data['questions'] = questions
                    write_file.write(json.dumps(data).encode() + b'\n')
                    status_file.write(f"{i}\n")
                    if i % 20 == 0:
                        write_file.flush()
                        status_file.flush()
                except Exception as e:
                    print(f"Error: {e}")
                    sys.stdout.flush()
                    continue

                #if count >= 100:
                #    break

        # os.remove(self.local_file_path)
        # os.remove(self.local_write_file_path)


def generate_queries(s3_file_path, batch_size, device_id, model_name, bucket_name, tmp_dir, worker_id):
    query_generation = QueryGeneration(s3_file_path, batch_size, device_id, model_name, bucket_name, tmp_dir, worker_id)
    query_generation.process()


def test():
    s3_file_path = 'open-assistant-retrieval/doc2query_pile/stage_0/pile-000.jsonl.gz'
    batch_size = 20
    device_id = 0
    model_name = 'doc2query/msmarco-t5-base-v1'
    bucket_name = 's-laion'
    tmp_dir = '/fsx/home-lkc/doc2query-experiments/tmp'
    worker_id = 0
    generate_queries(s3_file_path, batch_size, device_id, model_name, bucket_name, tmp_dir, worker_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--s3_file_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--model_name', type=str, default='doc2query/msmarco-t5-base-v1')
    parser.add_argument('--bucket_name', type=str, default='s-laion')
    parser.add_argument('--tmp_dir', type=str, default='/fsx/home-lkc/doc2query-experiments/tmp')
    parser.add_argument('--worker_id', type=int, default=0)
    args = parser.parse_args()

    generate_queries(args.s3_file_path, args.batch_size, args.device_id, args.model_name,
                     args.bucket_name, args.tmp_dir, args.worker_id)


if __name__ == '__main__':
    main()
    # test()

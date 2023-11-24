import multiprocessing
import os
import time
import torch
import json
import sys
import numpy as np
from data_utils.indexed_dataset import make_builder
from transformers import AutoTokenizer


# 1. Implement an Encoder, which gives it a line of input data and it returns you the tokenized result.
class Encoder(object):
    def __init__(self, model_path, max_prompt_length):
        self.max_prompt_length = max_prompt_length
        self.model_path = model_path

    def initializer(self):
        Encoder.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

    def encode(self, line):
        line = json.loads(line)
        if len(line["input"]) == 0:
            template = (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Response:\n"
            )
            prompt = template.format(instruction=line["instruction"])
        else:
            template = (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
            )
            prompt = template.format(
                instruction=line["instruction"], input=line["input"]
            )

        response = line["output"]
        prompt_tokens = Encoder.tokenizer.encode(prompt, add_special_tokens=False)
        full_tokens = Encoder.tokenizer.encode(
            prompt + response, add_special_tokens=False
        ) + [Encoder.tokenizer.eos_token_id]
        response_tokens = full_tokens[len(prompt_tokens) :]

        if len(prompt_tokens) > self.max_prompt_length:
            return None, None, None, None, len(line)

        return line, prompt, prompt_tokens, response_tokens, len(line)


def run(input_path, output_dir, only_prompt):
    # args = get_args()

    with open(input_path) as f:
        raw_data = f.readlines()

    dev_num = 500
    all_data = {
        "valid": raw_data[:dev_num],
        "train": raw_data[dev_num:],
    }

    for split in all_data:
        # encoder use the tokenizer to encode data
        encoder = Encoder(
            model_path="/nethome/cwan39/project/fl-minillm/checkpoint/gpt2-large",
            max_prompt_length=256,
        )

        # 2. Mapping all datas with Encoder, with the help of multiprocessing
        pool = multiprocessing.Pool(processes=32, initializer=encoder.initializer)
        encoded_docs = pool.imap_unordered(
            encoder.encode, all_data[split], chunksize=50
        )
        proc_start = time.time()
        total_bytes_processed = 0

        bin_file = os.path.join(output_dir, f"{split}_{0}.bin")
        idx_file = os.path.join(output_dir, f"{split}_{0}.idx")

        binary_builder = make_builder(bin_file, impl="mmap", dtype=np.uint16)

        # put tokenized data into binary_builder
        inst_num = 0
        print("#" * 10, split, "#" * 10)

        prompt_lens = []
        response_lens = []

        json_file = open(os.path.join(output_dir, f"{split}.jsonl"), "w")

        for lid, (line, prompt_str, prompt, response, bytes_processed) in enumerate(
            encoded_docs
        ):
            total_bytes_processed += bytes_processed
            if prompt is None:
                continue

            if only_prompt:
                if len(prompt) < 1024:
                    binary_builder.add_item(torch.IntTensor(prompt))
                else:
                    continue
            else:
                binary_builder.add_item(torch.IntTensor(prompt + [-1] + response))

            json_file.write(
                json.dumps(
                    {
                        "instruction": line["instruction"],
                        "prompt": prompt_str,
                        "input": line["input"],
                        "output": line["output"],
                    }
                )
                + "\n"
            )

            prompt_lens.append(len(prompt))
            response_lens.append(len(response))

            inst_num += 1
            if lid % 1000 == 0:
                current = time.time()
                elapsed = current - proc_start
                mbs = total_bytes_processed / elapsed / 1024 / 1024
                print(
                    f"Processed {lid} documents. {inst_num} instances.",
                    f"({lid/elapsed} docs/s, {mbs} MB/s).",
                    file=sys.stderr,
                )

        # finish compressing tokenized data into `bin_file`, and generate meta information into `idx_file`
        binary_builder.finalize(idx_file)

        # close multiproceessing mapping
        pool.close()
        json_file.close()

        print("Data num", len(prompt_lens))
        print(
            "Prompt lengths.",
            "Mean:",
            np.mean(prompt_lens),
            "Max:",
            np.max(prompt_lens),
            "Min:",
            np.min(prompt_lens),
        )
        print(
            "Response",
            "Mean:",
            np.mean(response_lens),
            "Max:",
            np.max(response_lens),
            "Min:",
            np.min(response_lens),
        )


def main():
    input_root_dir = "/data/cwan39/data/text/dolly"
    root_dir = "/data/cwan39/data/text/processed-dolly"

    for only_prompt in [True, False]:
        if only_prompt:
            cur_dir = os.path.join(root_dir, "prompt")
        else:
            cur_dir = os.path.join(root_dir, "full")

        input_path = os.path.join(input_root_dir, "public.jsonl")
        output_dir = os.path.join(cur_dir, "public")
        # output_path = os.path.join(output_dir, "public.jsonl")
        os.makedirs(output_dir, exist_ok=True)

        run(input_path, output_dir, only_prompt)

        for n_cluster in [1, 2, 4, 8]:
            for idx in range(n_cluster):
                print("now is processing", only_prompt, n_cluster, idx)
                input_path = os.path.join(
                    input_root_dir, f"{n_cluster}clients/private/data{idx}.jsonl"
                )
                output_dir = os.path.join(cur_dir, f"{n_cluster}clients/client{idx}")
                os.makedirs(output_dir, exist_ok=True)
                # output_path = os.path.join(output_dir, f"data{idx}.jsonl")

                run(input_path, output_dir, only_prompt)


if __name__ == "__main__":
    main()

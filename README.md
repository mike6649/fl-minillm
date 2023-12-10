# MiniLLM: Knowledge Distillation of Large Language Models

## Setup

### Environment

#### Hardware Dependencies

- At least two computation nodes, each of which is equipped with at least 4 Nvidia GPUs.

#### Software Dependencies

The main software used in this project are

- PyTorch 2.0.1
- DeepSpeed 0.10.1
- pynvml 11.5.0
- Datasets 2.14.6
- Accelerate 0.24.1

See `requirements.txt` for details.

### Installation

#### Option 1: Install with Pip

Running the following command will install all prerequisites from pip.

```bash
pip install -r requirements.txt
```

#### Option 2: Access Configured Environment

We are willing to allow you to access our configured environment. Please send us your public key for SSH access.

### Datasets

We use [`databricks-dolly-15k`](https://huggingface.co/datasets/databricks/databricks-dolly-15k) and [OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext) for evaluating this project. All datasets are prepared in the configured environment.

## Basic Usage

### Run Example Scripts

**Step 1: Allocate GPUs on PACE**

```bash
salloc -N2 -t3:00:00 --mem-per-gpu=12G --ntasks-per-node=2 --gpus-per-task=2
```

**Step 2: Dataset Processing**
1. The training/evaluation intruction-response data `dolly-databricks-15k` before processing can be downloaded from this [link](https://huggingface.co/datasets/databricks/databricks-dolly-15k).
1. The pre-training corpus can be downloaded from this [link](https://huggingface.co/datasets/Skylion007/openwebtext).
1. Process the pre-training corpus: `python3 tools/get_openwebtext.py`
1. Tokenize the two datasets and store them as binary files for easy access during training:
```bash
bash scripts/gpt2/tools/process_data_dolly.sh . # Process Dolly Train / Validation Data
bash scripts/gpt2/tools/process_data_pretrain.sh . # Process OpenWebText Train / Validation Data
```
The `dolly` dataset will be split into 4 parts during this step to allow FL training.

**Step 3: Run FL-MiniLLM Training**

```bash
srun bash fl/mpi_launch.sh
```

```bash
srun bash fl/baselines/fedavg.sh
```

**Step 4: Evaluation**

The below script will evaluate a model within the GPT2 family using the ROUGE-L score on the dolly dataset:
```bash
torchrun evaluate.py --json-data --model-path /path/to/the/model/
```

You can adjust the arguments listed in the above scripts to evaluate our project under different settings.

For the training and evaluation of the original MiniLLM script, please refer to [this repo](https://github.com/microsoft/LMOps/tree/main/minillm).

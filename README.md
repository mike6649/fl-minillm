# MiniLLM: Knowledge Distillation of Large Language Models

## Setup

### Environment

#### Hardware Dependencies

- A X86-CPU machine with at least GB host memory
- At least two computation nodes, each of which is equipted with at least 4 Nvidia GPUs.

#### Software Dependencies

The main softwares used in this projects are

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
salloc -N2 -t1:15:00 --mem-per-gpu=12G --ntasks-per-node=2 --gpus-per-task=2
```

**Step 2: Run FL-MiniLLM Training**

```bash
bash fl/mpi_launch.sh
```

**Step 3: Evaluation**

```bash
bash evaluate.sh
```

You can adjust the arguments listed in the above scripts for evaluating our project under different settings.

For the training and evaluation of the original MiniLLM script, please refer to [this repo](https://github.com/microsoft/LMOps/tree/main/minillm).

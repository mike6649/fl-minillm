import json
import torch
import os
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from sklearn.cluster import KMeans

private_ratio = 0.75
n_cluster_list = [1, 2, 4, 8, 16]

with open("/data/cwan39/data/text/minillm/dolly/raw.jsonl", "r") as jsonl_file:
    json_list = list(jsonl_file)

model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name, model_max_length=512)
model = BertModel.from_pretrained(model_name).cuda()

embd_list = []

for json_str in tqdm(json_list):
    item = json.loads(json_str)

    inputs = tokenizer(item["instruction"], return_tensors="pt", truncation=True).to(
        "cuda"
    )

    with torch.no_grad():
        embd = model(**inputs).last_hidden_state
        embd = embd.mean(1).view(-1)
    embd_list.append(embd.cpu().numpy())

np.random.seed(0)
idx_list = np.arange(len(json_list))
np.random.shuffle(idx_list)

public = []
for idx in tqdm(idx_list[int(len(idx_list) * private_ratio) :]):
    public.append(idx_list[idx])

os.makedirs(
    "/data/cwan39/data/text/dolly",
    exist_ok=True,
)

with open(
    "/data/cwan39/data/text/dolly/public.jsonl",
    "w",
) as f:
    for idx in public:
        f.write(json_list[idx])

for n_clusters in n_cluster_list:
    embd_list = np.array(embd_list)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embd_list)

    cluster_centers = kmeans.cluster_centers_
    embd_list = embd_list

    dist = torch.zeros(n_clusters)

    output_lists = [[] for _ in range(n_clusters)]

    # for i, json_str in tqdm(enumerate(json_list)):
    for idx in tqdm(idx_list[: int(len(idx_list) * private_ratio)]):
        # item = json.loads(json_str)
        embd = embd_list[idx]
        for j, cluster_center in enumerate(kmeans.cluster_centers_):
            # print(j)
            dist[j] = torch.norm(torch.from_numpy(cluster_center - embd))
            # print(dist[j])
        dist = torch.softmax(dist, dim=0)
        sel = np.random.choice(n_clusters, 1, p=dist.numpy())
        # output_lists[idx[0]].append(item)
        output_lists[sel[0]].append(idx_list[idx])

    os.makedirs(
        f"/data/cwan39/data/text/dolly/{n_clusters}clients/private",
        exist_ok=True,
    )

    for i, output_list in enumerate(output_lists):
        with open(
            f"/data/cwan39/data/text/dolly/{n_clusters}clients/private/data{i}.jsonl",
            "w",
        ) as f:
            for idx in output_list:
                f.write(json_list[idx])

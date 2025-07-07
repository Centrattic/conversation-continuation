# finding the good test prompt that I think could show meaningful growth over training, id = 228

import json
from datasets import load_dataset, DatasetDict, Dataset

# with open("test.json", "r", encoding="utf-8") as f:
#     data = json.load(f)

# for i in range(len(data)):
#     if "drop" in data[i]["prompt"]:
#         print(data[i]["prompt"])
#         print(i)


test_ds = load_dataset("json", data_files="test.json", split="train")
print(test_ds)
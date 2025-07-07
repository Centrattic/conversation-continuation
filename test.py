# finding the good test prompt that I think could show meaningful growth over training, id = 228

import json

with open("test.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for i in range(len(data)):
    if "drop" in data[i]["prompt"]:
        print(data[i]["prompt"])
        print(i)
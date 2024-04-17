import json

with open("/ssddata/weihao00/trl_codebase/data/math_metamathqa_395K.json", "r") as r:
    data_json = json.load(r)
import random


random.shuffle(data_json)



with open("/ssddata/weihao00/trl_codebase/data/sample_data", "w") as w:
    json.dump(data_json[:2560], w) 
print("bupt")
import copy
from alayalite import Client
import numpy as np
from multiprocessing import Pool
items = [
    (1, 1, np.array([0.1, 0.2, 0.3]), 1),
    (2, 2, np.array([0.4, 0.5, 0.6]), 2),
]
collection_name = "test_collection"
client = Client(".test.db")
if collection_name in client.list_collections():
    client.delete_collection(collection_name=collection_name, delete_on_disk=True)
collection = client.create_collection(
    collection_name
)
collection.insert(items)
client.save_collection(collection_name)

client = Client(".test.db")
collection = client.get_collection(collection_name)

params = {'vectors': [[0.1, 0.2, 0.3]], 'limit': 10}
params_lis = [copy.deepcopy(params) for _ in range(10)]
# multiple 100 times params

def wrap_search(params):
    print(params)
    return collection.batch_query(**params)
    

# 多进程搜索
with Pool(processes=1) as pool:
    res = pool.map(wrap_search, params_lis)
# res = collection.batch_query(vectors=[[0.1, 0.2, 0.3]], limit=2, ef_search=10, num_threads=1)
print(res)
# for i in range(10):
#     print(collection.batch_query(**params))
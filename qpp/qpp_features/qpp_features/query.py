from msgspec import Struct
from msgspec.json import decode

class Query(Struct):
    dist: str
    queryID2: str
    queryID1: str
    time:str

def load_json(path):
    data = None
    with open(path,'rb') as f:
        data = decode(f.read(), type=list[Query])
    return data


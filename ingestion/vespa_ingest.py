import requests

def index_document(doc: dict, dense: list, multi: list, sparse: list):
    url = f"http://vespa:8080/document/v1/vetsearch/vetsearch/docid/{doc['id']}"
    vespa_doc = {
        "fields": {
            "id": doc["id"],
            "contents": doc["contents"],
            "dense_embedding": dense,
            "multi_vector": {"cells": [{"address": {"x": str(i)}, "value": vec} for i, vec in enumerate(multi)]},
            "sparse_vector": {"cells": [{"address": {"x": item['term']}, "value": item['weight']} for item in sparse]},
            "course_id": doc["course_id"],
            "activity_id": doc["activity_id"],
            "strand": doc["strand"]
        }
    }
    return requests.post(url, json=vespa_doc)

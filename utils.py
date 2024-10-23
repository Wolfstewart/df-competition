import json
import requests

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from qdrant_client.models import PointStruct

from qdrant_client.models import Filter, FieldCondition, MatchValue
from config import *


def to_embedding(content):
    data = {
        "model": EMBEDDING_MODEL_NAME,
        "input": content
    }
    res = requests.post(EMBEDDING_API, json=data).json()
    datas = res['data']
    embeddings = [ele['embedding'] for ele in datas]
    return embeddings


class VECTOR:
    def __init__(self):
        self.client = self.connect_client()

    def connect_client(self):
        # 创建客户端
        client = QdrantClient(
            url="http://"+VECTOR_DB_HOST,
            port=PORT,
            grpc_port=GRPC_PORT,
            api_key=VECTOR_PASSWD
        )
        return client

    def rebuild_collection(self):
        print("初始化向量数据库：{}...".format(VECTOR_COLLECTION))
        self.drop_collection(VECTOR_COLLECTION)
        self.create_collection(VECTOR_COLLECTION)
        with open("dataset/rules1.json", 'r', encoding='utf-8') as reader:
            rules = json.loads(reader.read())
            start = 0
            batch = 8
            total_idx = 0
            while start < len(rules):
                print("current idx：{}".format(total_idx))
                batched_rules = rules[start: start+batch]
                contents = [rule['rule_text'] for rule in batched_rules]
                embeddings = to_embedding(contents)
                pts = []
                for rule, embedding in zip(batched_rules, embeddings):
                    pts.append(PointStruct(id=total_idx, vector=embedding, payload=rule))
                    total_idx += 1
                self.client.upsert(
                    collection_name=VECTOR_COLLECTION,
                    wait=True,
                    points=pts,
                )
                start += batch
        print("向量数据库初始化完成")

    # 创建索引
    def create_collection(self, collection):
        self.client.create_collection(
            # 设置索引的名称
            collection_name=collection,
            # 设置索引中输入向量的长度
            # 参数size是数据维度
            # 参数distance是计算的方法，主要有COSINE（余弦），EUCLID（欧氏距离）、DOT（点积），MANHATTAN（曼哈顿距离）
            vectors_config=VectorParams(size=EMBEDDING_DIMS, distance=Distance.COSINE),
        )

    def drop_collection(self,collection):
        self.client.delete_collection(collection_name=f"{collection}")

    def search(self, query, limit=5):
        embedding = to_embedding(query)[0]
        search_result = self.client.search(
            collection_name=VECTOR_COLLECTION,
            query_vector=embedding,
            limit=limit,
            search_params={"exact": False, "hnsw_ef": 128}, with_vectors=True
        )
        candidates = []
        for result in search_result:
            # print(result.payload)
            # print("")
            candidates.append(result.payload)
        return candidates


def to_json(data, out_file):
    with open(out_file, 'w', encoding='utf-8') as writer:
        json.dump(data, writer, ensure_ascii=False, indent=2)


def read_json(in_file):
    with open(in_file, 'r', encoding='utf-8') as reader:
        data = json.loads(reader.read())
    return data


if __name__ == '__main__':
    vector = VECTOR()
    # vector.rebuild_collection()
    vector.search("问题：在国家海洋局负责监测与应对海洋灾害的过程中，有一套详细的响应级别划分。考虑到风暴潮、海浪、海啸和海冰灾害的不同监测警报级别和应急响应等级，假设某日国家海洋局根据最新的气象数据和预测模型，面临了以下几种情况：1）预报中心发布了针对东海区域的风暴潮蓝色警报；2）连续2天针对南海的海冰情况发布了蓝色警报，并预计未来3天内预警区域的海冰情况将会持续恶化，可能对海上作业和航运造成重大影响；3）针对我国北部沿海区域发布了海浪灾害的蓝色警报。若国家海洋局根据其应急处置规则进行响应，下列哪项描述最准确地反映了这些情况下国家海洋局将会采取的行动？")



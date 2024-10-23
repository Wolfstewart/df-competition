import json
import os.path
import re

import requests

from config import *
from utils import *


"""
1、 规则数据向量化保存数据库
2、 问题数据向量化召回最接近的topk
3、 处理结果（答案、参考规则）

"""

PROMPT = """
现在进行阅读理解任务，根据题干信息和参考信息，从给出的ABCD选项中选出正确的答案，只返回选项字母，不返回其他内容！示例如下：
示例1：
参考信息：Ⅰ级应急响应应急加密观测：海浪灾害影响期间，受影响海区的分局组织开展海浪加密观测工作。海浪自动观测点全天24小时加密观测；海浪人工观测点每日8时至17时加密观测。加密观测指令由分局统一下达。海冰灾害影响期间，北海分局组织相关中心站和海洋站每日开展1次重点岸段现场巡视与观测，并在每日14时前将数据通过电子邮件发送预报中心和受影响的省（自治区、直辖市）海洋预报机构；同时协调飞机每周开展2次航空遥感观测，并在飞机降落12小时内，由北海航空支队通过电子邮件传输至北海预报中心，经北海预报中心汇总后，传输至预报中心和受影响的省（自治区、直辖市）海洋预报机构。　　海啸灾害影响期间，预报中心利用全球地震监测预警系统密切监视地震、海啸发生发展动态，并及时从中国地震台网中心、太平洋海啸警报中心、西北太平洋海啸预警中心获取海啸预警信息。
问题：在应对海冰灾害过程中，北海分局组织相关中心站和海洋站对受影响的重点岸段进行巡视与观测。这项措施的执行频率和结果上报流程对于及时有效的灾害应急管理至关重要。请问，根据规则，这些巡视与观测活动的执行频率是多少，以及观测数据上报的时间要求是什么？同时，请指出航空遥感观测的执行频率及其数据上报的时限。
选择：A.每日开展1次巡视与观测，数据在每日14时前通过电子邮件上报。航空遥感观测每周2次，数据在飞机降落12小时内上报。 B.每日开展2次巡视与观测，数据在每日16时前通过电子邮件上报。航空遥感观测每周3次，数据在飞机降落24小时内上报。 C.每日开展1次巡视与观测，数据在每日12时前通过电子邮件上报。航空遥感观测每周1次，数据在飞机降落24小时内上报。 D.每日开展2次巡视与观测，数据在每日18时前通过电子邮件上报。航空遥感观测每周2次，数据在飞机降落18小时内上报。
示例1答案：A

参考信息：{}
{}
请回答：
"""


def get_rule_map():
    info = {}
    records = read_json('dataset/rules1.json')
    for record in records:
        info[record["rule_id"]] = record['rule_text']
    return info


def get_test_map():
    info = {}
    records = read_json('dataset/rules1.json')
    for record in records:
        info[record["question_id"]] = record['question_text']
    return info


def chat(query):
    data = {
        "model": LLM_MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": query
            }
        ],
        "temperature": 0.2      # 0.9
    }
    res = requests.post(LLM_API, json=data).json()
    ans = res['choices'][0]['message']['content']
    return ans


def clean(s: str):
    li = list(s)
    for ele in li:
        ele = ele.upper()
        if ele in list("ABCD"):
            return ele
    return "A"      # 蒙A


# 题干信息全部用来做向量检索。检索结果未重新排序直接用于答案生成。
def ori_process(datas):
    """
    题干信息全部用来做向量检索。检索结果未重新排序直接用于答案生成。
    :param datas:
    :return:
    """
    answers = []
    vector = VECTOR()
    for data in datas:
        question_id = data['question_id']
        question = data['question_text']
        rules = vector.search(question, 10)
        context = "\n".join([rule['rule_text'] for rule in rules])
        rule_ids = [str(rule['rule_id']) for rule in rules]
        prompt = PROMPT.format(context, question)
        # print(prompt)
        ans = chat(prompt)
        ans = clean(ans)
        print("question id: {}, answer: {}, rule id: {}".format(question_id, ans, rule_ids))
        answers.append({"question_id": question_id, "answer": ans, "rule_id": rule_ids})
    with open("submit.json", 'w', encoding='utf-8') as writer:
        json.dump(answers, writer, ensure_ascii=False, indent=2)
    to_json(answers, "output/submit.json")


def rerank_process(datas):
    """
    题干信息全部用来做向量检索。检索结果重新排序后组合让大模型生成答案，返回的rule_id也是重排序之后的序号。
    :param datas:
    :return:
    """
    answers = []
    vector = VECTOR()
    for data in datas:
        question_id = data['question_id']
        question = data['question_text']
        # print(question)
        rules = vector.search(question, 10)
        context = "\n".join([rule['rule_text'] for rule in rules])
        rule_ids = [str(rule['rule_id']) for rule in rules]
        prompt = PROMPT.format(context, question)
        # print(prompt)
        ans = chat(prompt)
        ans = clean(ans)
        print("question id: {}, answer: {}, rule id: {}".format(question_id, ans, rule_ids))
        answers.append({"question_id": question_id, "answer": ans, "rule_id": rule_ids})
    to_json(answers, "output/submit.json")


# 直接使用题干信息，检索出前20条规则。
def retrieve_naive(datas):
    retrieve_result = {}
    vector = VECTOR()
    for data in datas:
        question_id = data['question_id']
        question = data['question_text']
        rules = vector.search(question, 20)
        retrieve_result[question_id] = rules
    to_json(retrieve_result, "output/retrieve-top20.json")
    return retrieve_result


# 不带答案选项的问题进行查询
def retrieve_without_choice(datas):
    retrieve_result = {}
    vector = VECTOR()
    for data in datas:
        question_id = data['question_id']
        question = data['question_text']
        rules = vector.search(question, 20)
        retrieve_result[question_id] = rules
    to_json(retrieve_result, "output/retrieve-without-abcd-top20.json")
    return retrieve_result



# 增加召回的方案， 先召回20条，重排序后取前10位。先用conan向量模型，查看重排序的结果和召回的顺序是否不一致。
def ra_process(testdata):
    rule_map = get_rule_map()
    q_info = get_test_map()
    retrievefile = "output/retrieve-top20.json"
    if os.path.isfile(retrievefile):
        retrievedata = read_json(retrievefile)
        print("检索结果从本地读取完毕。。。")
    else:
        retrievedata = retrieve_naive(testdata)
        print("检索结果从向量库中检索完毕。。。")

    # 重排序前后对比，使用cos
    for q_id, rules in retrievedata.items():
        print(q_id, rules)
        pre_orders = [rule['rule_id'] for rule in rules]
        question_text = q_info[q_id]
        rule_contents = [rule['rule_text'] for rule in rules]
        query_embed = to_embedding(question_text)[0]
        rule_embeds = to_embedding(rule_contents)


        break


if __name__ == '__main__':
    testdata = read_json('dataset/test1.json')

    # 直接RAG
    # ori_process(testdata)

    # 检索
    ra_process(testdata)
    # answers = []
    # vector = VECTOR()
    # patern = "问题：(.*?)[\n]*(选择)[\n]*.*"
    # for data in datas:
    #     # print(data)
    #     question_id = data['question_id']
    #     question = data['question_text']
    #     print(question_id)
    #     query = re.findall(patern, question)[0]
    #     # rules = vector.search(query, 10)
    #     print(query)
    #     # context = "\n".join([rule['rule_text'] for rule in rules])
    #     # rule_ids = [str(rule['rule_id']) for rule in rules]
    #     # prompt = PROMPT.format(context, question)
    #     # # print(prompt)
    #     # ans = chat(prompt)
    #     # print("question id: {}, answer: {}, rule id: {}".format(question_id, ans, rule_ids))
    #     # answers.append({"question_id": question_id, "answer": ans, "rule_id": rule_ids})

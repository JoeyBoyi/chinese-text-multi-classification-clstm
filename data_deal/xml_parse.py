# -*- coding: utf-8 -*-

import xmltodict
import re
from xml.dom.minidom import Document
import mmap


def regulation(string):
    string = re.sub(r"[^\u4e00-\u9fa5A-z0-9！？，。\s\.\!\/`_,$%^*(+\"\')\<\>\/.,\"=:_-\{\}-、，－、；\\［］;\-（）：。《》〉+——()?“”！，;；<=>～~{|}。？、~@#￥……&*（）]+", "", string)
    return string


with open('/home/xw/codeRepository/NLPspace/QA/jira_issue_rcnn_multi_clf/data/2017-Dec-21--1828/entities.xml','r', encoding='utf-8') as fr:
    with mmap.mmap(fr.fileno(), 0 , access=mmap.ACCESS_READ) as m:
        data = xmltodict.parse(m.read(), encoding='utf-8')

print ("-"*40+"I am here"+"-"*40)
issue_list = data['entity-engine-xml']['Issue']
project_list = data['entity-engine-xml']['Project']
issuetype_list = data['entity-engine-xml']['IssueType']
action_list = data['entity-engine-xml']['Action']

print ("-"*40+"I am here"+"-"*40)
# Project节点 id: name 映射表
project_dict = dict()
for pro in project_list:
    project_dict[pro["@id"]] = pro["name"]

# IssueType节点 id: name 映射表
issuetype_dict = dict()
for it in issuetype_list:
    issuetype_dict[it["@id"]] = it["name"]

# Action节点 issue: body 映射表
comments_dict = dict()
for com in action_list:
    if "@body" in  action_list:
        comments_dict[com["@issue"]] = com["@body"]
    else:
        comments_dict[com["@issue"]] = com["body"]
        
# 创建dom文档
doc = Document()

# 创建根节点
root_list = doc.createElement('list')
# 根节点插入dom树
doc.appendChild(root_list)

# 依次将orderDict中的每一组元素提取出来，创建对应节点并插入dom树
for issue in issue_list:
    ID_name = issue["@id"]
    
    project_id = issue["@project"]
    project_name = project_dict[project_id]
    
    type_id = issue["@type"]
    issuetype_name = issuetype_dict[type_id]
    
    summary_name = issue["@summary"]
    
    if "@description" in issue:
        description_name = issue["@description"]
    else:
        description_name = issue["description"]
    
    comments_name = comments_dict[ID_name]
    
    

    # 每一组信息先创建节点<Issue>，然后插入到父节点<list>下
    Issue = doc.createElement('Issue')
    root_list.appendChild(Issue)

    # 将ID插入<Issue>中
    # 创建节点<id>
    ID = doc.createElement('id')
    # 创建<id>下的文本节点
    ID_text = doc.createTextNode(ID_name)
    # 将文本节点插入到<id>下
    ID.appendChild(ID_text)
    # 将<id>插入到父节点<Issue>下
    Issue.appendChild(ID)

    # 将project插入<Issue>中，处理同上
    project = doc.createElement('project')
    project_text = doc.createTextNode(project_name)
    project.appendChild(project_text)
    Issue.appendChild(project)

    # 将issuetype插入<Issue>中，处理同上
    issuetype = doc.createElement('issuetype')
    issuetype_text = doc.createTextNode(issuetype_name)
    issuetype.appendChild(issuetype_text)
    Issue.appendChild(issuetype)

    # 将summary插入<Issue>中，处理同上
    summary = doc.createElement('summary')
    summary_text = doc.createTextNode(summary_name)
    summary.appendChild(summary_text)
    Issue.appendChild(summary)

    # 将description插入<Issue>中，处理同上
    description = doc.createElement('description')
    description_text = doc.createTextNode(description_name)
    description.appendChild(description_text)
    Issue.appendChild(description)

    # 将comments插入<Issue>中，处理同上
    comments = doc.createElement('comments')
    comments_text = doc.createTextNode(comments_name)
    comments.appendChild(count_text)
    Issue.appendChild(comments)

# 将dom对象写入本地xml文件
with open("/home/xw/codeRepository/NLPspace/QA/jira_issue_rcnn_multi_clf/data/jira_issue_data.xml", 'w') as f:
    f.write(doc.toprettyxml(indent='  ', encoding='utf-8'))


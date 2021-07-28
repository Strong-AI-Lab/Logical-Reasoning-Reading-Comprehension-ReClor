# -*- coding: utf-8 -*-
"""
Created on Wed March 3 2021

@author: Qiming Bao

Depth=2 new data generation for negation rule

"""

import json
import itertools
import random

people_name = ['Anne', 'Alan', 'Bob', 'Charlie', 'Dave', 'Erin', 'Harry', 'Gary', 'Fiona']
people_relations_1 = ['is', 'is not']

people_attributes_1 = ['big', 'strong', 'high', 'huge']
people_attributes_2 = ['short', 'thin', 'small', 'little']

people_attributes_3 = ['wealthy', 'smart', 'nice', 'quiet', 'kind']
people_attributes_4 = ['poor', 'dull', 'rough', 'bad', 'sad']

people_attributes_5 = ['old']
people_attributes_6 = ['young']

id = 0
whole_dict = []
item = list(itertools.permutations(people_name, 4))
for index in range(0, len(item)):
    id += 1
    people = item[index][0]
    people_1 = item[index][1]
    people_2 = item[index][2]
    people_3 = item[index][3]
    random.shuffle(people_attributes_1)
    random.shuffle(people_attributes_2)
    random.shuffle(people_attributes_3)
    random.shuffle(people_attributes_4)
    random.shuffle(people_attributes_5)
    random.shuffle(people_attributes_6)
    context = [people + " " + people_relations_1[0] + " " + people_attributes_1[0] + ". " +
               people + " " + people_relations_1[0] + " " + people_attributes_1[1] + ". " +
               people_1 + " " + people_relations_1[0] + " " + people_attributes_2[0] + ". " +
               people_1 + " " + people_relations_1[0] + " " + people_attributes_2[1] + ". " +
               people_2 + " " + people_relations_1[0] + " " + people_attributes_3[0] + ". " +
               people_3 + " " + people_relations_1[0] + " " + people_attributes_4[0] + ". " +
               people_3 + " " + people_relations_1[0] + " " + people_attributes_4[1] + ". " +
               "If someone is not " + people_attributes_1[2] + " then they are " + people_attributes_4[0] + ". " +
               "If someone is not " + people_attributes_4[2] + " then they are " + people_attributes_3[3] + ". " +
               "If someone is " + people_attributes_3[0] + " then they are " + people_attributes_3[1] + ". " +
               "If someone is " + people_attributes_3[1] + " and not " + people_attributes_4[3] + " then they are " +
               people_attributes_3[2] + ". " +
               "If someone is " + people_attributes_4[0] + " and not " + people_attributes_1[2] + " then they are " +
               people_attributes_4[4] + ". " +
               "If someone is " + people_attributes_2[0] + " and " + people_attributes_2[1] + " then they are " +
               people_attributes_4[3] + ". " +
               "If someone is " + people_attributes_4[3] + " and not " + people_attributes_3[1] + " then they are " +
               people_attributes_4[1] + ". " +
               "All " + people_attributes_3[3] + " people are " + people_attributes_3[4] + ". "
               ]
    question0 = people + " " + people_relations_1[0] + " " + people_attributes_4[4] + "."
    label0 = "true"
    QDep0 = "2"
    question0_0 = people + " " + people_relations_1[1] + " " + people_attributes_4[4] + "."
    label0_0 = "false"
    QDep0_0 = "2"
    question1 = people_3 + " " + people_relations_1[0] + " " + people_attributes_3[4] + "."
    label1 = "true"
    QDep1 = "2"
    question1_1 = people_3 + " " + people_relations_1[1] + " " + people_attributes_3[4] + "."
    label1_1 = "false"
    QDep1_1 = "2"
    question2 = people_2 + " " + people_relations_1[0] + " " + people_attributes_3[2] + "."
    label2 = "true"
    QDep2 = "2"
    question2_2 = people_2 + " " + people_relations_1[1] + " " + people_attributes_3[2] + "."
    label2_2 = "false"
    QDep2_2 = "2"
    question3 = people_1 + " " + people_relations_1[0] + " " + people_attributes_4[1] + "."
    label3 = "true"
    QDep3 = "2"
    question3_3 = people_1 + " " + people_relations_1[1] + " " + people_attributes_4[1] + "."
    label3_3 = "false"
    QDep3_3 = "2"

    test_dict = {
        'id': "NegationRule-D2-" + str(id),
        'context': context[0],
        'questions': [{
            'id': "NegationRule-D2-" + str(id) + "1",
            'text': question0,
            'label': label0,
            'meta': {
                "QDep": QDep0,
                "QCat": "0_not_notTrue"
            }},
            {'id': "NegationRule-D2-" + str(id) + "2",
             'text': question0_0,
             'label': label0_0,
             'meta': {
                 "QDep": QDep0_0,
                 "QCat": "0_0_not_notTrue"
             }},
            {'id': "NegationRule-D2-" + str(id) + "3",
             'text': question1,
             'label': label1,
             'meta': {
                 "QDep": QDep1,
                 "QCat": "0_not_notTrue"
             }},
            {'id': "NegationRule-D2-" + str(id) + "4",
             'text': question1_1,
             'label': label1_1,
             'meta': {
                 "QDep": QDep1_1,
                 "QCat": "0_0_not_notTrue"
             }},
            {'id': "NegationRule-D2-" + str(id) + "5",
             'text': question2,
             'label': label2,
             'meta': {
                 "QDep": QDep2,
                 "QCat": "0_true_trueNot"
             }},
            {'id': "NegationRule-D2-" + str(id) + "6",
             'text': question2_2,
             'label': label2_2,
             'meta': {
                 "QDep": QDep2_2,
                 "QCat": "0_0_true_trueNot"
             }},
            {'id': "NegationRule-D2-" + str(id) + "7",
             'text': question3,
             'label': label3,
             'meta': {
                 "QDep": QDep3,
                 "QCat": "0_true_trueNot"
             }},
            {'id': "NegationRule-D2-" + str(id) + "8",
             'text': question3_3,
             'label': label3_3,
             'meta': {
                 "QDep": QDep3_3,
                 "QCat": "0_0_true_trueNot"
             }}]
    }

    whole_dict.append(test_dict)


def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError


with open('NegationRule-D2.jsonl', 'w') as f:
    for index in whole_dict:
        json.dump(index, f, default=set_default)
        f.write('\n')

#from __future__ import printfunction
import sys
import pickle
import codecs
from collections import defaultdict as mydict

def printSets(entity_setlist, namelist):
    name = set()
    for item in entity_setlist:
        for it in item:
            name.add(it)
    print('Name\t',end='')
    for item in name:
        print(item+'\t',end='')
    print('')
    for names,entity in zip(namelist,entity_setlist):
        print(names+'\t',end='')
        for it in name:
            print('%d\t' % len(entity[it]),end='')
        print('')

test_path = str(sys.argv[1])

Entityset = mydict(set)
EntityDetectset = mydict(set)
EntityCorrectset = mydict(set)

Entitylist = mydict(list)
EntityDetectlist = mydict(list)
EntityCorrectlist = mydict(list)

flag_0 = False
flag_1 = False
flag_2 = False

entity_name = ''
entity_detect_name = ''
entity_correct_name = ''

with codecs.open(test_path, 'r', encoding='utf-8') as file_r:
    for lines in file_r.readlines():
        line = lines.strip()
        item = line.split('\t')
        if(len(item) < 3):
    	    continue
        true_label = item[-2].strip()
        mark_label = item[-1].strip()
        word = item[0].strip()
        #统计真实的情况
        if('B_' in true_label):
            entity_name = word
            flag_0 = True
        elif('I_' in true_label and flag_0):
                entity_name += word
        elif('E_' in true_label and flag_0):
            entity_name += word
            Entityset[true_label].add(entity_name)
            Entityset['Total'].add(entity_name)
            Entitylist[true_label].append(entity_name)
            Entitylist['Total'].append(entity_name)
            entity_name = ''
            flag_0 = False
        else:
            entity_name = ''
            flag_0 = False

            #统计检测到的情况
        if('B_' in mark_label):
            entity_detect_name = word
            flag_1 = True
        elif('I_' in mark_label and flag_1):
            entity_detect_name += word
        elif('E_' in mark_label and flag_1):
            entity_detect_name += word
            EntityDetectset[mark_label].add(entity_detect_name)
            EntityDetectlist[mark_label].append(entity_detect_name)
            EntityDetectset['Total'].add(entity_detect_name)
            EntityDetectlist['Total'].append(entity_detect_name)
            entity_detect_name = ''
            flag_1 = False
        else:
            entity_detect_name = ''
            flag_1 = False

        #统计正确的情况
        if(true_label == mark_label):
            if('B_' in true_label):
                flag_2 = True
                entity_correct_name = word
            elif('I_' in true_label and flag_2):
                entity_correct_name += word
            elif('E_' in true_label and flag_2):
                entity_correct_name += word
                EntityCorrectset[true_label].add(entity_correct_name)
                EntityCorrectlist[true_label].append(entity_correct_name)
                EntityCorrectset['Total'].add(entity_correct_name)
                EntityCorrectlist['Total'].append(entity_correct_name)
                entity_correct_name = ''
            else:
                entity_correct_name = ''
                flag_2 = False
        else:
            entity_correct_name = ''
            flag_2 = False


print('In terms of set')
printSets([Entityset,EntityDetectset,EntityCorrectset],['True','Detect','Correct'])
print('In terms of list')
printSets([Entitylist,EntityDetectlist,EntityCorrectlist],['True','Detect','Correct'])
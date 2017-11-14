#-*- coding: utf-8 -*-
import pynlpir
import re
input_files = ["./ie_employee/LabeledData.1.txt", "./ie_employee/LabeledData.2.txt", "./ie_employee/LabeledData.3.txt", "./ie_employee/LabeledData.4.txt", "./ie_employee/LabeledData.5.txt"]
pos_file = "./ie_employee/POS.txt"
infos = []

for input_file in input_files:
    with open(input_file, 'r', encoding='ANSI') as instream:
        infos += list(map(lambda x:x.strip(), instream.readlines()))

pynlpir.open(encoding='utf_8')

with open(pos_file, 'w', encoding='utf-8') as outstream:
    for sentence in infos:
        if '|' in sentence:
            continue
        if sentence != "":
            p = re.compile(r'\{(.*?)/(\w\w)\}')
            names = set(p.findall(sentence))
            p_p = p.sub(r'\1',sentence)
            segments = pynlpir.segment(p_p)
            for i in range(len(segments)):
                segments[i] = list(segments[i])
                segments[i].append('O')
            for i, seg in enumerate(segments):
                #print(seg)
                for length in range(1,5):
                    if i+length-1 >= len(segments):
                        break
                    test_name = ""
                    for k in range(i,i+length):
                         test_name += segments[k][0]
                    # print(length, test_name)
                    for name in names:
                        if test_name == name[0]:
                            
                            segments[i][2] = "B_"+name[1]
                            for k in range(i+1,i+length-1):
                               segments[k][2] = "I_"+name[1]
                            if i+length-1 != i:
                                segments[i+length-1][2] = "E_"+name[1]

            
            for segment in segments:
                outstream.write(segment[0])
                outstream.write('\t')
                # 这是个bug，它有时会把noun识别不出来，返回None
                if segment[1] is None:
                    outstream.write("noun")
                else:
                    outstream.write(segment[1])
                outstream.write('\t')
                outstream.write(segment[2])
                outstream.write("\n")
        outstream.write("\n")

pynlpir.close()
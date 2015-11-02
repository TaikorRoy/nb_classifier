# -*- coding:utf-8 -*-

from lib.baseClass import NB
from py_utility import system


if __name__ == "__main__":
    model_path = "model/nb_model.model"
    verification_samples_path = "data/verification_set.txt"
    verification_samples = system.get_content_list(verification_samples_path)

    n_verification_samples = len(verification_samples)

    global_counter = 0
    cate_counter = dict()
    precision_counter = dict()
    # {"互联网":[33, 87], "体育":[14, 53] .....  }

    for ele in verification_samples:
        ele_list = ele.split("\t")
        text = ele_list[0]
        label = ele_list[1]
        model = system.json_loads(model_path)
        cl = NB(model["prior_table"], model["posterior_table"])
        result = cl.predict(text)
        print(result)

        if label not in cate_counter.keys():
            cate_counter[label] = [0, 0]

        if result not in precision_counter.keys():
            precision_counter[result] = [0, 0]

        if result == label:
            global_counter += 1
            cate_counter[label][0] += 1
            cate_counter[label][1] += 1
            precision_counter[result][0] += 1
            precision_counter[result][1] += 1
        else:
            cate_counter[label][1] += 1
            precision_counter[result][1] += 1

    precision = float(global_counter)/float(n_verification_samples) * 100
    print("Global Precision: " + "\t" + str(precision) + "%")
    print("Global Detail:" + "\t" + str(global_counter) + "/" + str(n_verification_samples))

    print("--------------------------------------")

    print("Cate Recall:")
    for key in cate_counter.keys():
        cate_recall = float(cate_counter[key][0])/float(cate_counter[key][1]) * 100
        str_part1 = key + "\t" + str(cate_counter[key][0]) + "/" + str(cate_counter[key][1])
        str_part2 = "\t" + str(cate_recall) + "%"
        excel_str = str_part1 + "\t" + str_part2
        print(excel_str)

    print("--------------------------------------")

    print("Cate Precision:")
    for key in precision_counter.keys():
        cate_precision = float(precision_counter[key][0])/float(precision_counter[key][1]) * 100
        str_part1 = key + "\t" + str(precision_counter[key][0]) + "/" + str(precision_counter[key][1])
        str_part2 = "\t" + str(cate_precision) + "%"
        excel_str = str_part1 + "\t" + str_part2
        print(excel_str)

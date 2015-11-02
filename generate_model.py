# -*- coding:utf-8 -*-

from lib.baseClass import model_generator
from py_utility import system

if __name__ == "__main__":
    category_list_path = "data/category_list.txt"
    training_set_path = "data/training_set.txt"
    model_path = "model/nb_model.model"

    model = model_generator(category_list_path, training_set_path)
    prior_table, posterior_table = model.train()
    model = {"prior_table": prior_table, "posterior_table": posterior_table}
    system.json_dumps(model_path, model)

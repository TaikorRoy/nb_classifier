__author__ = 'Taikor'

from functools import reduce
from py_utility import system
import pynlpir


class NB:
    def __init__(self, prior_table, posterior_table):
        # df_table  {"valid":{PoS1:df1, PoS2:df2,...},"invalid":{PoS1:df1, PoS2:df2,...}}
        # prior_table {"valid":X, "invalid":Y}
        tmp_cate_list = prior_table.keys()
        temp = [prior_table[cate] for cate in tmp_cate_list]
        temp_sum = sum(temp)
        prior = dict()   # construct a prior dictionary
        for ele in tmp_cate_list:
            prior[ele] = float(prior_table[ele])/float(temp_sum)

        self.posterior_table = posterior_table
        self.prior_table = prior_table
        self.category_list = tmp_cate_list
        self.prior = prior

    def comp_prop(self, category_key, words_set):   # P(A/B) * P(B)
        word_prob_list = list()
        for word in words_set:
            if word in self.posterior_table[category_key].keys():
                temp_v = self.posterior_table[category_key][word]
                temp_pv = self.prior_table[category_key]
                word_prob = float(temp_v)/float(temp_pv)    # the probability of the a word x appear in any position of the given article
            else:
                word_prob = 1/float(self.prior_table[category_key])
                # expand 20 times, so that very small prob number can be avoided
            word_prob = 20 * word_prob
            word_prob_list.append(word_prob)
        posterior = reduce(lambda x, y: x*y, word_prob_list)
        prior = self.prior[category_key]
        prob = posterior*prior
        return prob

    def predict(self, text):
        # words = [word1, word2, word3, ...]
        pynlpir.open()
        seg_words = pynlpir.segment(text, pos_tagging=False)
        words_set = set(seg_words)
        result = dict()
        for category in self.category_list:
            prob = self.comp_prop(category, words_set)
            result[category] = prob

        """
        buffer = [result[my_key] for my_key in result.keys()]
        score_sum = sum(buffer)
        # result = {my_key: result[my_key]/score_sum for my_key in result.keys()}
        """
        buffer = list(result.items())
        buffer.sort(key=lambda x: x[1], reverse=True)
        top_category = buffer[0][0]
        return top_category


class model_generator:
    def __init__(self, category_list_path, training_set_path):
        category_list = system.get_content_list(category_list_path)
        training_set_material = system.get_content_list(training_set_path)
        self.category_list = category_list
        self.training_set_material = training_set_material

    def train(self):
        # df_table = {"valid": {"science": 35, "physics": 34, "robot": 57}, "invalid": {"fat": 30, "large": 34, "cheap": 55}}
        # The number of articles containing "science", "physics" or "robot"
        # prior_table = {"valid": 183, "invalid": 244}
        pynlpir.open()
        prior_table = {ele: 0 for ele in self.category_list}
        posterior_table = {ele: dict() for ele in self.category_list}

        i = 0
        for sample in self.training_set_material:
            buffer = sample.split("\t")
            text = buffer[0]
            seg_words = pynlpir.segment(text, pos_tagging=False)
            words_set = set(seg_words)
            try:
                label = buffer[1]
            except:
                print("Line " + str(i) + "in training set corrupted")
                continue
            prior_table[label] += 1
            for word in words_set:   # all words in the text
                if word in posterior_table[label].keys():
                    posterior_table[label][word] += 1   # posterior count +1 when this word already exists in posterior
                else:
                    posterior_table[label][word] = 1  # posterior count assigned to 1 when this word does exist in posterior yet
            i += 1
        return prior_table, posterior_table



# coding:utf-8
import pre_deal_data.data_deal as pd
import collections
import file_operation as fo


class WordsContent(object):

    def __init__(self, max_volume):
        self.data = None
        self.unknown = 'UNK'
        self.end_char = 'END'
        self.vocabulary_size = max_volume
        self.words_ids = dict()
        self.ids_words = dict()
        self.words_frp = dict()
        self.label_words_ids = dict()
        self.label_ids_words = dict()

    def line_size(self):
        return len(self.data)

    def label_size(self):
        return len(self.label_words_ids)

    def create_data_content(self, input_data):
        count = [[self.unknown, -1], [self.end_char, -1]]
        total_word_dict = pd.statistic_word_from_line(input_data)
        count.extend(collections.Counter(total_word_dict).most_common(self.vocabulary_size - 2))
        for word, frp in count:
            self.words_ids[word] = len(self.words_ids)
            self.words_frp[word] = frp
        unk_count = 0
        data = list()
        for line in input_data:
            line_data = list()
            for word in line:
                if word in self.words_ids:
                    index = self.words_ids[word]
                else:
                    index = 0
                    unk_count += 1
                line_data.append(index)
            data.append(line_data)

        self.data = data
        self.words_frp[self.unknown] = unk_count
        self.ids_words = dict(zip(self.words_ids.values(), self.words_ids.keys()))

    def create_label_content(self, input_data):
        count = [[self.unknown, -1], [self.end_char, -1]]
        total_word_dict = pd.statistic_word_from_line(input_data)
        count.extend(collections.Counter(total_word_dict).most_common())
        for word, frp in count:
            if word != '':
                self.label_words_ids[word] = len(self.label_words_ids)
        self.label_ids_words = dict(zip(self.label_words_ids.values(), self.label_words_ids.keys()))

    def add_content(self, input_data):
        assert len(self.words_ids) > 0
        data = list()
        for line in input_data:
            line_data = list()
            for word in line:
                if word in self.words_ids:
                    index = self.words_ids[word]
                else:
                    index = 0
                line_data.append(index)
            data.append(line_data)
        self.data = data

    def words_to_ids(self, words):
        return [self.words_ids[w] for w in words if w in self.words_ids]

    def ids_to_words(self, ids):
        return [self.ids_words[index] for index in ids if index in self.ids_words]

    def label_words_to_ids(self, words):
        return [self.label_words_ids[w] for w in words if w in self.label_words_ids]

    def label_ids_to_words(self, ids):
        return [self.label_ids_words[index] for index in ids if index in self.label_ids_words]

    def add_words(self, words):
        for w in words:
            if w in self.words_ids:
                self.words_frp[w] += 1
            else:
                index = len(self.words_ids)
                self.words_ids[w] = index
                self.ids_words[index] = w
        return self

    def load_word_index(self, path):
        self.words_ids = fo.load_tuple_text_from_file(path)
        self.ids_words = dict(zip(self.words_ids.values(), self.words_ids.keys()))
        return self

    def save_word_index(self, path):
        indeies = sorted(self.words_ids.items(), key=lambda x: x[1])
        fo.write_tuple_to_text(path, indeies)

    def load__label_word_index(self, path):
        self.label_words_ids = fo.load_tuple_text_from_file(path)
        self.label_ids_words = dict(zip(self.label_words_ids.values(), self.label_words_ids.keys()))
        return self

    def save_label_word_index(self, path):
        indeies = sorted(self.label_words_ids.items(), key=lambda x: x[1])
        fo.write_tuple_to_text(path, indeies)

    def data_reshape_to_one(self):
        long_data = []
        end_id = self.words_ids[self.end_char]
        for line in self.data:
            line.append(end_id)
            long_data.extend(line)
        self.data = long_data
        return self


def create_content_data(data, vocabulary_size, save_path=None):
    wc = WordsContent(vocabulary_size)
    wc.create_data_content(data)
    if save_path is not None:
        wc.save_word_index(save_path)
    return wc


def create_content_from_file(load_path, vocabulary_size):
    wc = WordsContent(vocabulary_size)
    wc.load_word_index(load_path)
    return wc



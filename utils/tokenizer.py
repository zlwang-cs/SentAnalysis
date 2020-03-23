import nltk


class ParaSplit:
    def __init__(self):
        self.tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        pass

    def split(self, para):
        para = para.lower()
        # return para.split('.')
        return self.tokenizer.tokenize(para)

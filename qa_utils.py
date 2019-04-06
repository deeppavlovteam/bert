import re
import json
import pickle

import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords


class QADoc:
    def __init__(self, doc):
        self.question = doc['subject']
        self.question_extension = doc['content']
        self.best_answer = doc['bestanswer']
        self.answers = [a for a in doc['nbestanswers'] if a != self.best_answer]

    @property
    def all_answers(self):
        answers = [self.best_answer] + self.answers
        answers = self.filter_none(answers)
        return answers

    @staticmethod
    def filter_none(texts):
        return list(filter(lambda x: x is not None, texts))


class YahooL6QBERTDataset:
    def __init__(self, docs, tokenizer=None, tf_idf_max_features=2 ** 19, use_bigram=True):

        # Read a json file if needed
        if isinstance(docs, str):
            with open(docs) as f:
                docs = json.load(f)

        if tokenizer is None:
            print('Warning! Using regex tokenizer!')
        self.tokenizer = tokenizer

        self.docs = [QADoc(doc) for doc in docs]

        if use_bigram:
            ngram_range = (1, 2)
        else:
            ngram_range = (1,)

        self._question_words = ['what', 'when', 'where', 'which', 'who', 'whom', 'whose', 'why', 'how']

        self.tfidf = TfidfVectorizer(tokenizer=self._tokenize,
                                     ngram_range=ngram_range,
                                     stop_words=stopwords.words('english') + ['\n', '\n\n'] + self._question_words,
                                     max_features=tf_idf_max_features)

        self.answers, self.doc_indices = self._get_all_answers()

    def fit_tfidf(self, use_questions=True, use_answers=True):
        texts = []
        if use_answers:
            texts.extend(self.answers)
        if use_questions:
            texts.extend(self._get_all_questions())

        print('Fitting TF-IDF')
        self.tfidf.fit(texts)

        print('Transforming answers')
        self._answers_tfidf_vectors = self.tfidf.transform(self.answers)

    def _get_all_questions(self, docs=None, provide_qestion_extentions=True):
        if docs is None:
            docs = self.docs
        questions = []

        questions.extend([doc.question for doc in docs])
        if provide_qestion_extentions:
            questions.extend([doc.question_extension for doc in docs])
        questions = list(filter(lambda x: x is not None, questions))
        return questions

    def _get_all_answers(self, docs=None):
        if docs is None:
            docs = self.docs
        doc_indices = []
        answers = []
        for n, doc in enumerate(docs):
            for ans in doc.all_answers:
                if ans is not None:
                    doc_indices.append(n)
                    answers.append(ans)

        return answers, doc_indices

    def _tokenize(self, s):
        if self.tokenizer is not None:
            return self.tokenizer.tokenize(s)
        else:
            return re.findall("[\w']+|[^\w ]", s)

    def _is_question(self, q):
        q_lower = q.lower()
        if '?' in q or any(w in q_lower for w in self._question_words):
            return True
        else:
            return False

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'docs': self.docs,
                         'answers_tfidf_matrix': self._answers_tfidf_vectors,
                         'answers': self.answers}, f)

    def load(self, path):
        with open(path, 'rb') as f:
            params_dict = pickle.load(f)
            self.docs = params_dict['docs']
            self._answers_tfidf_vectors = params_dict['answers_tfidf_matrix']
            self.answers = params_dict['answers']

    def sample_qa_pairs(self,
                        false_answer_rate=0.5,
                        not_best_answer_rate=0.3,
                        tfidf_top_k=1000,
                        shuffle=True,
                        question_extension_rate=0.0):
        if shuffle:
            order = np.random.permutation(len(self.docs))
        else:
            order = np.arange(len(self.docs))
        for ind in tqdm(order):
            doc = self.docs[int(ind)]
            q = doc.question
            qe = doc.question_extension
            question = self.choose_between_question_and_extension(q, qe, question_extension_rate)
            if np.random.rand() < false_answer_rate:
                ans = self.sample_answer(question, tfidf_top_k, exclude_doc_ind=ind)
                mark = 0
            else:
                pick_other_answer = doc.best_answer is None
                pick_other_answer |= len(doc.answers) > 0 and np.random.rand() > not_best_answer_rate
                if not pick_other_answer:
                    ans = doc.best_answer
                else:
                    ans = np.random.choice(doc.answers)
                mark = 1
            yield question, ans, mark

    def choose_between_question_and_extension(self, q, qe, question_extension_rate):
        if qe is None:
            return q
        elif q is not None:
            if q == qe:
                return q

            is_q = self._is_question(q)
            is_qe = self._is_question(q)

            if is_q and not is_qe:
                return q
            if is_qe and not is_q:
                return qe

            if question_extension_rate and np.random.rand() < question_extension_rate:
                valid_extension = self._is_question(qe)
                valid_extension &= len(q) < len(qe)
                if valid_extension:
                    return qe
            return q
        elif q is None:
            if self._is_question(qe):
                return qe

    def sample_answer(self, question, top_k, exclude_doc_ind=None):
        q_vec = self.tfidf.transform([question])
        simmilaryty = self._answers_tfidf_vectors.dot(q_vec.T).toarray().squeeze(1)
        top_inds = np.argsort(simmilaryty)[-top_k:]
        ans_ind = top_inds[np.random.randint(top_k)]
        if exclude_doc_ind:
            while ans_ind == exclude_doc_ind:
                ans_ind = top_inds[np.random.randint(top_k)]
        return self.answers[ans_ind]



import numpy as np
import nltk
import gensim
import numpy.random as nprnd


class Word2Vec:
    def __init__(self):
        """
            Creates a word2vec helper given the dimension of a word vector
        """
        # Load Google's pre-trained Word2Vec model.
        self.model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin',
                                                                     binary=True)
        # if we cannot find the word in pre-trained model, using a near-zero random vector
        # to represent the missing information. We do not want to use a fixed value (e.g. zero)
        # because that incorrectly implies all missing words are the same word
        self.unknowns = np.random.uniform(-0.01, 0.01, 300).astype("float32")

    def get(self, word):
        """
            Given a word, return its vector representation
        :param word:
        :return:
        """
        if word not in self.model.vocab:
            return self.unknowns
        else:
            return self.model.word_vec(word)


class DataFeeder:
    def __init__(self, mode, sentence_length_cap=100, model_base_path="./WikiQA_Corpus/WikiQA-"):
        """
        :param sentence_length_cap: truncate sentences longer than this limit
        """
        self.questions, self.answers, self.labels = [], [], []

        # in our BCNN model, we also allow a few manually crafted feature to be used in the final layer
        # these features are currently computed at data loading time
        self.engineered_features = []
        self.engineered_feature_count = 0
        self.sentence_length_cap = sentence_length_cap
        self.data_path = model_base_path
        self.mode = mode
        self.data_size = 0
        assert mode == 'train' or mode == 'test'
        self.word2vec = Word2Vec
        self.max_sentence_length = 0

    def load_data(self):
        """
            Load data into memory
        :return:
        """
        with open(self.data_path + self.mode + ".txt", "r", encoding="utf-8") as f:
            stopwords = nltk.corpus.stopwords.words("english")
            self.max_sentence_length = 0
            for line in f:
                items = line[:-1].split("\t")

                question = items[0].lower().split()

                # Truncate answers because some of them can get really long.
                # Brutal truncation may not work for some domains.
                # TODO - figure out a better way to limit computation.
                answer = items[1].lower().split()[:self.sentence_length_cap]

                label = int(items[2])
                if label > 0:  # if it's 1, it's a correct answer
                    self.questions.append(question)
                    self.answers.append(answer)

                    # count number of overlapping non-stop words
                    word_cnt = len([word for word in question if (word not in stopwords) and (word in answer)])

                    #
                    self.engineered_features.append([len(question), len(answer), word_cnt])

                    local_max_len = max(len(question), len(answer))
                    if local_max_len > self.max_sentence_length:
                        self.max_sentence_length = local_max_len

        print "Max sentence length:{}".format(local_max_len)
        assert len(self.questions) == len(self.answers), "there must be equal number of questions and answers"
        assert len(self.questions) == len(self.labels), "there must be equal number of questions and labels"

        self.data_size = len(self.questions)

        # compute additional features based on IDF (inverse document frequency)
        def flatten(l):
            return [token for tokenized_sentence in l for token in tokenized_sentence]

        answer_vocabulary = set(flatten(self.answers))
        idf = {}
        for word in answer_vocabulary:
            idf[word] = np.log(self.data_size / len([1 for answer in self.answers if word in answer]))

        for i in range(self.data_size):
            weighted_word_cnt = sum([idf[word] for word in self.questions[i]
                                     if (word not in stopwords) and (word in answer_vocabulary)])
            self.engineered_features[i].append(weighted_word_cnt)

        self.engineered_feature_count = len(self.engineered_features[0])

    def has_more(self):
        """
        Has the batch iterator run out?
        """
        return self.index < self.data_size

    def convert2tensor(self, tokenized_sentence):
        """
            Convert a tokenized sentence into a tensor
            1. every word (i.e. token) becomes a column vector
            2. pad 0s to the end of the sentence so sentences have equal length
            3. expand 1 dimension at the front, so we can concatenate individual
               sentences into a mega tensor
        :param tokenized_sentence:
        :return: a tensor of dimensions [1, word_vector_length, sentence_length]
        """
        padded_sentence = np.pad(np.column_stack([self.word2vec.get(w) for w in tokenized_sentence]),
                                 [[0, 0], [0, self.max_sentence_length - len(tokenized_sentence)]],
                                 "constant")
        return np.expand_dims(padded_sentence, axis=0)

    def next_batch(self, batch_size):
        batch_size = min(self.data_size - self.index, batch_size)
        question_tensors, answer_tensors = [], []

        for i in range(batch_size):
            s1 = self.questions[self.index + i]
            s2 = self.answers[self.index + i]

            question_tensors.append(self.convert2tensor(s1))
            answer_tensors.append(self.convert2tensor(s2))

        # [batch_size, word_vector_length, sentence_length]
        combined_question_tensor = np.concatenate(question_tensors, axis=0)
        combined_answer_tensor = np.concatenate(answer_tensors, axis=0)
        label_vector = self.labels[self.index:self.index + batch_size]
        batch_engineered_features = self.engineered_features[self.index:self.index + batch_size]

        self.index += batch_size

        return combined_question_tensor, combined_answer_tensor, label_vector, batch_engineered_features

    @staticmethod
    def reset_seed(seed):
        nprnd.seed(seed)

    @staticmethod
    def multiplex_training_pairs(questions, answers, sample_size):
        """
            Given questions and answers pairs, generate training pairs and labels.
            In our Q&A settings, our input only has pairs of (question, correct_answer), i.e.,
            we do not have negative sets. To solve this problem, we generate negative training
            samples by randomly sample all possible answers.
            For example, given:
            (q1, a1) ... (q2, a2), ...
            We create the following training data:
            (q1, a1, 1) # correct answer from input
            (q1, a4, 0) # randomly generated wrong answer
            (q1, a5, 0)
            ......
            (q2, a2, 1)
            (q2, a1, 0)
        :param questions: a list of question objects (we don't care the underlying data type of question,
                as long as it is a list
        :param answers: a list of answer objects
        :param sample_size: number of answers randomly sampled in addition to the input
        :return:
        """
        assert len(questions) == len(answers)
        question_count = len(questions)
        expanded_questions, expanded_answers, labels = [], [], []
        for i in range(len(questions)):
            # append the positive pair first
            expanded_questions.append(questions[i])
            expanded_answers.append(answers[i])
            labels.append(1)

            # now append the randomly generated negative pairs
            idx = nprnd.randint(0, question_count, sample_size)
            expanded_questions.extend([questions[i] for j in idx])
            expanded_answers.extend([answers[j] for j in idx])
            labels.extend([1 if j == i else 0 for j in idx])

        return expanded_questions, expanded_answers, labels

import unittest
from preprocess import DataFeeder


class DataFeederTestCase(unittest.TestCase):
    def test_negative_sample_creation(self):
        DataFeeder.reset_seed(123)
        questions, answers, labels = DataFeeder.multiplex_training_pairs(['q1', 'q2', 'q3', 'q4', 'q5'],
                                                                         ['a1', 'a2', 'a3', 'a4', 'a5'],
                                                                         sample_size=2)
        self.assertEqual(questions, ['q{}'.format(i+1) for i in range(5) for j in range(3)])
        self.assertEqual(answers, ['a1', 'a3', 'a5',
                                   'a2', 'a3', 'a2',
                                   'a3', 'a4', 'a3',
                                   'a4', 'a4', 'a2',
                                   'a5', 'a2', 'a1'])
        self.assertEqual(labels, [1, 0, 0,
                                  1, 0, 1,
                                  1, 0, 1,
                                  1, 1, 0,
                                  1, 0, 0])

if __name__ == '__main__':
    unittest.main()

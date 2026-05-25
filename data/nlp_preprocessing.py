import torch
import torch.nn as nn
from torchtyping import TensorType

# torch.tensor(python_list) returns a Python list as a tensor
class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        vocabulary = set()
        all_sentences = positive + negative
        for sentence in all_sentences:
            for word in sentence.split():
                vocabulary.add(word)
        sorted_list = sorted(list(vocabulary))

        mapping = {}
        index = 1
        for word in sorted_list:
            mapping[word] = index
            index += 1

        output_lists = []
        for sentence in all_sentences:
            integers = []
            for word in sentence.split():
                integers.append(mapping[word])
            output_lists.append(torch.tensor(integers))

        return nn.utils.rnn.pad_sequence(output_lists, batch_first=True)


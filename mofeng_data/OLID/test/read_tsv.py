from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import sys
import os
import json
import random
import logging
import numpy as np
import six
from io import open
from collections import namedtuple

from ernie import tokenization
from ernie.batching import pad_batch_data

log = logging.getLogger(__name__)

if six.PY3:
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


def csv_reader(fd, delimiter='\t'):
    def gen():
        for i in fd:
            yield i.rstrip('\n').split(delimiter)

    return gen()


class ReadTsvTest:
    def __init__(self):
        self.for_cn = True

    def read_tsv(self, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, 'r', encoding='utf8') as f:
            reader = csv_reader(f)
            headers = next(reader)
            text_indices = [
                index for index, h in enumerate(headers) if h != "label"
            ]
            Example = namedtuple('Example', headers)
            print(Example)
            examples = []
            for line in reader:
                for index, text in enumerate(line):
                    if index in text_indices:
                        if self.for_cn:
                            line[index] = text.replace(' ', '')
                        else:
                            line[index] = text

                example = Example(*line)
                examples.append(example)
            return examples


if __name__ == "__main__":
    tsv = ReadTsvTest()
    result = tsv.read_tsv("../task_a/train.tsv")
    # print(result)

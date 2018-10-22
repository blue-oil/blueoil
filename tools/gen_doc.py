from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

from tools.gen_doc_augmentor import main as gen_doc_augmentor_main


def main(docs_dir):
    gen_doc_augmentor_main(docs_dir)


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    docs_dir = os.path.join(base_dir, 'docs')
    main(docs_dir)

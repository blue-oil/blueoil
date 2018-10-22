from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import inspect
import os
import sys
import re

sys.path.insert(0, "lmnet")

from lmnet.data_processor import Processor
import lmnet.data_augmentor as augmentor


def main(docs_dir):
    def write_doc(doc):
        doc = "`" + doc[:doc.find(":")] + "`" + doc[doc.find(":"):]
        doc_file.write(doc)
        doc_file.write('\n\n')

    with open(os.path.join(docs_dir, 'reference/augmentation.md'), 'w') as doc_file:
        doc_file.write('Data Augmentation\n')
        doc_file.write('======\n\n')

        for name, obj in inspect.getmembers(augmentor):
            if not inspect.isclass(obj) or not issubclass(obj, Processor):
                continue
            doc = obj.__doc__
            doc_file.write('#### {}\n\n'.format(name))
            doc_file.write(doc[:doc.find("Args")].strip())

            args = doc[doc.find("Args:\n") + 6:doc.find("Returns")].strip()
            if args:
                args.replace("\n", "\n\n")
                doc_file.write('\n\n_params_:\n\n')
                last_arg = 0
                for m in re.finditer(r'[a-z].* \([a-z].*\):', args):
                    if last_arg == 0:
                        pass
                    else:
                        write_doc(args[last_arg:m.start(0)].strip())
                    last_arg = m.start(0)
                write_doc(args[last_arg:].strip())
            doc_file.write("\n\n")


if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    docs_dir = os.path.join(base_dir, 'docs')
    main(docs_dir)

# How to document DLK

For DLK, we should write documents along with your development.  

## File Structure (only around document files)
```
  dlk  
  └──docs  
     ├──cpp  
     │  └──Doxyfile
     ├──python
     │  └──docs
     │     ├──index.rst
     │     └──Makefile
     └──Makefile
```

## Comments in source code
You should comment at the header of each functions/methods you wrote.

### Python
The format of Python comments must follow `PEP257` or **"docstrings"**. Multiline docstrings with descriptions of parameters / return values would be appreciated.<br>  
For details, see [the official page](https://www.python.org/dev/peps/pep-0257/).

To check whether your comments follow the rule, you can use `pydocstyle`. To install it, try:
```
pip install pydocstyle
```
To check, just do:
```
pydocstyle <file or directory>
```

### C++
For C++ code, we use `doxygen`.

So the documentation, please follow [the official page](http://www.doxygen.jp/docblocks.html) (in Japanese).

## Other documents
You can freely add documents under `python\docs` directory, or modify `index.rst`. You can use **".rst" (reStructuredText)** or **".md" (Markdown)** files.

## Generation of documents
In the top-level directory, just do:
```
make
```
For cleaning up all the generated documents, try:
```
make clean
```

For Python documents, see `docs/python/_build/html/index.html` with your browser.<br>
For C++ documents, see `docs/cpp/html/index.html` with your browser.


## Type Hints
We recommend you to add type hints to the signatures of your functions/methods.<br>
See [the official page](https://docs.python.org/3.6/library/typing.html).

In the near future, types will be checked automatically in automatic testing with CI.

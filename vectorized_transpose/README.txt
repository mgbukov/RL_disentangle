1. .pyx file contains the cython functions: the interface btw python and cpp

2. setup.py file contains the cython compiler instructions (links to any libraries, header files, cpp compiler arguments, etc.)

3. ./cpp_headers/ contains the cpp header files

4. test.py calls the wrapped functions in python


To compile the code using cython, use

```
python setup.py build_ext -i

```


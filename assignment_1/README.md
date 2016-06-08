# CMPE561: Natural Language Processing
## Assignment 1
Naive Bayes classifier for Turkish texts.

Download and extract the *69 Authors* dataset from [here](http://www.kemik.yildiz.edu.tr/?id=28), which was the primary dataset for this assignment.

The usages for each class can be found in the comments. In general, you should only need to run the Preprocessor and the Tester.
If need be, you can use the provided classes, such as Tokenizer and Naive Bayes implementations.

To run the preprocessor, use

    ./preprocessor.py path/to/dataset path/to/training/set path/to/test/set
This will split your dataset to training and test.

To run the tester, use

    ./tester.py path/to/training/set path/to/test/set
This will output the results of the classifiers to the console. Note that only the outputs of Bag of Words feature set and the Bag of Character N-Grams feature set are displayed. You can read the [report](Report.ipynb) on how the other feature sets perform.

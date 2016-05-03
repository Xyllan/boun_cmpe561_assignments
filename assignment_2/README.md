# CMPE561: Natural Language Processing
## Assignment 2
HMM POS Tagger.

The usages for each little program can be found in their own comments. The metu_sabanci_cmpe_561 folder is not necessary for running the codes, but it is necessary to run the codes in the Jupyter Notebook report. Simply extract and place it in this folder.

The CoNLL Parser parses the given CoNLL file into sentences, each a list of word tuples. There is limited use case for this program outside the methods it provides, but if need be, it can return a list of cpostags using

    ./conll_parser.py path/to/conll/file.conll
The HMM PoS Tagger Trainer can be run according to the format found in the assignment description:

    ./train_hmm_tagger.py path/to/training/file.conll --cpostag
or

    ./train_hmm_tagger.py path/to/training/file.conll --postag
The first option will use the cpostags in creating the training configuration of the HMM, the second will use the postags. Both will create a file named `hmm.conf` that contains the configuration of the HMM in JSON format.

To tag a file (which is assumed to be of roughly CoNLL format), use the `hmm_tagger` program as

    ./hmm_tagger.py path/to/test/file path/to/output/file.txt
This will tag each word of the sentence will the best PoS tag estimate. The output file has a format `word|Tag` as taken from the sample output.

To evaluate the results of the above program with the gold standard, use the `evaluate_hmm_tagger` program by calling

    ./evaluate_hmm_tagger.py path/to/output/file.txt path/to/gold/standard.conll
This will output the accuracies for all of the tags, plus the overall accuracy. It will also print the list of tags and the resulting confusion matrix. Note that the second argument must be of the CoNLL format.
You can read the [report](Report.ipynb) for the results.
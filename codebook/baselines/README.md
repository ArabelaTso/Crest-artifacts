To run SIT and PatInv, change directory to where StanfordCoreNLP located, and run:

`java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -preload tokenize,ssplit,pos,lemma,ner,parse -status_port 9001 -port 9001`


The file `synonyms.dat` is used for `PatInv.py`

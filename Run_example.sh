#!/bin/sh
################################################################
# Example of running an experiment on the command line
################################################################

# These assume that the English EmoInt data are in directory ../emoint_ei-en relative to the package root directory
echo "------"
date
echo "Assuming EmoInt data is in directory ../emoint_ei-en"


#  TRAINING
echo "------"
date
echo "Training..."
java -cp 'slda.jar:lib/*' cmd.CmdTSLDA -v ../emoint_ei-en/ei-en-anger-vocab.txt -tp ../emoint_ei-en/ei-en-anger-hac-ld.txt -d ../emoint_ei-en/ei-en-anger-corpus-train.txt -l ../emoint_ei-en/ei-en-anger-labels-train.txt -m MODEL_ei-en-anger.model

#  TESTING
echo "------"
date
echo "Testing..."
java -cp 'slda.jar:lib/*' cmd.CmdTSLDA -v ../emoint_ei-en/ei-en-anger-vocab.txt -tp ../emoint_ei-en/ei-en-anger-hac-ld.txt -d ../emoint_ei-en/ei-en-anger-corpus-test.txt -l ../emoint_ei-en/ei-en-anger-labels-test.txt -m MODEL_ei-en-anger.model -t -p PREDICTIONS.txt 

#  EVALUATION
echo "------"
date
echo "Computing correlation..."
java -cp 'slda.jar:lib/*' cmd.CmdEval -p PREDICTIONS.txt -l ../emoint_ei-en/ei-en-anger-labels-test.txt

# CLEANUP
echo "------"
date
echo "To clean up:"
echo "rm -i {MODEL,PREDICTIONS}*"



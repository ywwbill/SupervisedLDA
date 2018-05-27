# <h1 id="top">Supervised LDA</h1>A package of [supervised LDA](#slda_ref) which can incorporate labels, [tree priors](#tree_prior_ref), and hinge loss. Most code can be applied to general data, as long as they meet the format requirements. The code for computing Pearson correlation coefficient is specifically implemented for the [EmoInt data](https://competitions.codalab.org/competitions/17751).* [Dependencies](#dependencies)* [Use the Tools in Command Line](#command)	* [Supervsied LDA](#slda)	* [Supervised LDA with Tree Priors](#tslda)	* [Building Tree priors](#tree_prior)	* [Evaluating the Correlation](#pearson)* [Resources](#resources)* [References](#refs)## <h2 id="dependencies">Dependencies</h2>- Java 8.- `slda.jar` in the root directory.- The `lib/` directory.## <h2 id="command">Use Supervised LDA Tools in Command Line</h2>The general command line format is```java -cp slda.jar:lib/* cmd.{Tools} -arg1 <arg1-value> -arg2 <arg2-value> ... -argn <argn-value>```- **<font size=4>Windows users please replace `slda.jar:lib/*` with `slda.jar;lib/*`.</font>**- You may add extra JVM options as needed, e.g., `-Xmx20G` to request a maximum of 20GB memory if your dataset is large.- The options for `{Tools}` are	- [`CmdSLDA`](#slda): Run supervised LDA.	- [`CmdTSLDA`](#tslda): Run supervised LDA with tree priors.	- [`CmdTree`](#tree_prior): Build tree priors using pre-trained word embeddings.	- [`CmdEval`](#pearson): Compute the Pearson correlation coefficient between predictions and gold labels. This is implemented specifically for the [EmoInt data](https://competitions.codalab.org/competitions/17751).- For each tool, you can always use `-h` option to get help information.### <h3 id="slda">Supervised LDA</h3>```java -cp slda.jar:lib/* cmd.CmdSLDA -v <vocab-file> -d <corpus-file> -l <label-file> -m <model-file>```- Required arguments	- `<vocab-file>`: Vocabulary file. Each line contains a unique word.	- `<corpus-file>`: Corpus file in which documents are represented by word indexes and frequencies. Each line contains a document in the following format			```		<doc-len> <word-type-1>:<frequency-1> <word-type-2>:<frequency-2> ... <word-type-n>:<frequency-n>		```			`<doc-len>` is the total number of *tokens* in this document. `<word-type-i>` denotes the i-th word in `<vocab-file>`, starting from 0. Words with zero-frequency can be omitted.	- `<model-file>`: Trained model file in JSON format. Read and written by program.- Semi-optional arguments	- `-l <label-file>` [optional in test]: Label file. Each line contains the corresponding document's numeric label. If a document's label is not available, leave the corresponding line empty.- Optional arguments	- `-t`: Use the model for test (default: false).	- `-a <alpha-value>`: Parameter of the Dirichlet prior of document distributions over topics (default: 0.01). Must be a positive real number.	- `-b <beta-value>`: Parameter of the Dirichlet prior of topic distributions over words (default: 0.01). Must be a positive real number.	- `-k <num-topics>`: Number of topics (default: 20). Must be a positive integer.	- `-i <num-iters>`: Number of iterations (default: 500). Must be a positive integer.	- `-mu <mu-value>`: The mean of the Gaussian priors for regression parameters (default 0.0).	- `-n <nu-value>`: The variance of the Gaussian priors for regression parameters (default: 1.0). Must be a positive real number.	- `-s <sigma-value>`: The variance for the Gaussian distribution for generating documents' response labels (default: 1.0). Must be a positive real number.	- `-hl`: Use hinge loss as the loss function (default false).	- `-c <c-value>`: The regularization parameter for hinge loss (default 1.0). Must be a positive real number.	- `-e <epsilon-value>`: The error bound for hinge loss (default 0.1). Must be a positive real number.	- `-tc <topic-count-file>`: File for documents' topic counts. Each line contains a document's numbers of tokens assigned to topics. Topic counts are separated by space.	- `-r <topic-file>`: File for showing human-readable topics and top positive/negative words.	- `-w <num-top-word>`: Number of top words for human-readable topics and for positive/negative weights (default: 20). Must be a positive integer.	- `-p <pred-file>`: File for predicted values. Each line contains a predicted value.### <h3 id="tslda">Supervised LDA with Tree Priors</h3>```java -cp slda.jar:lib/* cmd.CmdTSLDA -v <vocab-file> -tp <tree-prior-file> -d <corpus-file> -l <label-file> -m <model-file>```- Arguments are the same with [supervised LDA](#slda), except for one more required:	- `-tp <tree-prior-file>`: File of tree priors. Tree priors can be built using the [tree prior construction tool](#tree_prior) and pre-trained word embeddings. Or you can build your own following the [format](https://stackoverflow.com/a/1649223). The representation of a leaf node is `<word-id>:<word>` where `<word-id>` is word's ID, i.e., the line number of this word in the `<vocab-file>` (starting from 0), and `<word>` is the string representation of the word itself.### <h3 id="tree_prior">Building Tree Priors</h3>```java -cp slda.jar:lib/* cmd.CmdTree -v <vocab-file> -e <embedding-file> -o <tree-prior-file>```- Required arguments	- `-v <vocab-file>`: Vocabulary file. Same format with [supervised LDA](#slda).	- `-e <embedding-file>`: Pre-trained word embedding file. Follows the format of word2vec output: The first line contains the numbers of words and dimensions, separated by space; Each of the following line contains the word and its embeddings, separated by space.	- `-o <tree-prior-file>`: The file for storing human-readable tree priors.- Optional arguments	- `-t <tree-prior-type>`: The type of [tree priors](#tree_prior_ref) (default 1):		- 1: Two-level tree prior.		- 2: Hierarchical agglomerative clustering.		- 3: Hierarchical agglomerative clustering with leaf duplication.	- `-k <child-number>`: The number of child nodes per internal node for a two-level tree (default 10). Must be a positive integer.### <h3 id="pearson">Evaluating the Correlation</h3>```java -cp slda.jar:lib/* cmd.CmdEval -p <prediction-file> -l <gold-label-file>```- Compute and print the Pearson correlation coeffients of predictions and gold labels of	- All examples	- The examples with gold labels greater than 0.5- The `<prediction-file>` and `<gold-label-file>` must have the same number of lines.- This is specifically implemented for [EmoInt task](https://competitions.codalab.org/competitions/17751). Values are not reliable for other tasks.- Required arguments	- `-p <prediction-file>`: The predicted value file. Each line contains a predicted value. Can be written by the `-p <pred-file>` option in [supervised LDA](#slda).	- `-l <gold-label-file>`: The gold label file. Format same with `<prediction-file>`.- Optional arguments	- `-o <output-file>`: The file for writing the two Pearson correlation coefficients. The first line contains the Pearson correlation coefficient of all examples. The second line contains the Pearson correlation coefficient of the examples with gold labels greater than 0.5.## <h2 id="resources">Resources</h2>- [EmoInt shared task](https://competitions.codalab.org/competitions/17751)- [Pre-trained English tweet embeddings](http://nlp.stanford.edu/data/glove.twitter.27B.zip)	- The embeddings are trained using GloVe. The file does not the number of words and dimensions in its first line, contrast to the requirements in [tree prior](#tree_prior) construction.- [Pre-trained Spanish tweet embeddings](http://4530.hostserv.eu/resources/embed_tweets_es_200M_200D.zip)- Processed EmoInt data: /fs/clip-lorelei/wwyang/EmoInt/shared-task/	- Includes the EmoInt task data for English and Spanish. File names tell what they are. File formats follow the requirements above.- Unlabeled data: /fs/clip-lorelei/wwyang/EmoInt/text/	- The file names denote the date, language, and emotion.	- Each line of each file contains a tweet ID and its corresponding tweet which includes at least one of the [keywords](http://saifmohammad.com/WebDocs/AIT-2018/SemEval2018-Task1-QueryTerms.zip), separated by tab.	- The data is collected daily automatically.	- The data contains retweets (which start with RT) that should be filtered.## <h2 id="refs">References</h2>### <h3 id="slda_ref">[Supervised LDA](#slda)</h3>Jon D. McAuliffe and David M. Blei. 2008. Supervised topic models. In Proceedings of Advances in Neural Information Processing Systems.### <h3 id="tree_prior_ref">[Tree Priors](#tree_prior)</h3>Weiwei Yang, Jordan Boyd-Graber, and Philip Resnik. 2017. Adapting Topic Models using Lexical Associations with Tree Priors. In Proceedings of Empirical Methods in Natural Language Processing.
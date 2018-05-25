package tlda;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.FileInputStream;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

import cc.mallet.util.Randoms;
import tlda.util.TLDADoc;
import tlda.util.TLDATopicNode;
import tlda.util.TLDAWord;
import util.MathUtil;
import util.format.Fourmat;
import util.IOUtil;

import com.google.gson.Gson;

public class TLDA
{	
	public static final int TRAIN=0;
	public static final int TEST=1;
	
	public final TLDAParam param;
	
	protected static Randoms randoms;
	protected static Gson gson;
	
	protected double alpha[];
	protected double updateDenom;

	protected int numDocs;
	protected int numWords;
	protected int numTestWords;
	protected final int type;
	
	protected ArrayList<TLDADoc> corpus;
	protected TLDATopicNode topics[];
	protected ArrayList<ArrayList<ArrayList<TLDATopicNode>>> topicNodeMaps;
	protected double theta[][];
	
	protected double logLikelihood;
	protected double perplexity;
	
	public void readCorpus(String corpusFileName) throws IOException
	{
		readCorpus(corpusFileName, true);
	}
	
	public void readCorpus(String corpusFileName, boolean indexed) throws IOException
	{
		BufferedReader br=new BufferedReader(new InputStreamReader(new FileInputStream(corpusFileName), "UTF-8"));
		String line;
		while ((line=br.readLine())!=null)
		{
			if (indexed)
			{
				corpus.add(new TLDADoc(line, param.numTopics, param.numVocab, param.topicPrior.getNumLeafNodes()));
			}
			else
			{
				corpus.add(new TLDADoc(line, param.numTopics, param.vocabMap, param.topicPrior.getNumLeafNodes()));
			}
		}
		br.close();
		numDocs=corpus.size();
	}
	
	protected void printParam()
	{
		IOUtil.println("Running "+this.getClass().getSimpleName());
		IOUtil.println("\t#docs: "+numDocs);
		IOUtil.println("\t#tokens: "+numTestWords);
		param.printBasicParam("\t");
	}
	
	public void initialize()
	{
		initializeTopicNodeMaps();
		initDocVariables();
		initTopicAssigns();
		if (param.verbose) printParam();
	}
	
	public void initializeTopicNodeMaps()
	{
		topicNodeMaps=new ArrayList<ArrayList<ArrayList<TLDATopicNode>>>();
		for (int topic=0; topic<param.numTopics; topic++)
		{
			topicNodeMaps.add(new ArrayList<ArrayList<TLDATopicNode>>());
			for (int vocab=0; vocab<param.numVocab; vocab++)
			{
				topicNodeMaps.get(topic).add(new ArrayList<TLDATopicNode>());
				for (int path=0; path<param.vocabPaths.get(vocab).size(); path++)
				{
					topicNodeMaps.get(topic).get(vocab).add(topics[topic].getNode(param.vocabPaths.get(vocab).get(path)));
				}
			}
		}
	}
	
	protected void initTopicAssigns()
	{
		for (TLDADoc doc : corpus)
		{
			int interval=getSampleInterval();
			for (int token=0; token<doc.docLength(); token+=interval)
			{
				int topic=randoms.nextInt(param.numTopics);
				int word=doc.getWord(token);
				int path=randoms.nextInt(topicNodeMaps.get(topic).get(word).size());
				doc.assignTopicAndNode(token, topic, topicNodeMaps.get(topic).get(word).get(path));
			}
		}
	}
	
	protected void initDocVariables()
	{
		updateDenom=0.0;
		numWords=0;
		for (int doc=0; doc<numDocs; doc++)
		{
			numWords+=corpus.get(doc).docLength();
			int sampleSize=getSampleSize(corpus.get(doc).docLength());
			updateDenom+=(double)(sampleSize)/(double)(sampleSize+param.alpha*param.numTopics);
		}
		theta=new double[numDocs][param.numTopics];
		getNumTestWords();
	}
	
	public void sample(int numIters)
	{
//		computeLogLikelihood();
//		perplexity=Math.exp(-logLikelihood/numTestWords);
//		IOUtil.println("Perplexity: "+perplexity);
		for (int iteration=1; iteration<=numIters; iteration++)
		{
			for (int doc=0; doc<numDocs; doc++)
			{
				sampleDoc(doc);
			}
			computeLogLikelihood();
			perplexity=Math.exp(-logLikelihood/numTestWords);
			if (param.verbose)
			{
				IOUtil.println("<"+iteration+">"+"\tLog-LLD: "+format(logLikelihood)+"\tPPX: "+format(perplexity));
			}
			if (param.updateAlpha && iteration%param.updateAlphaInterval==0 && type==TRAIN)
			{
				updateHyperParam();
			}
		}
		
		if (type==TRAIN && param.verbose)
		{
			for (int topic=0; topic<param.numTopics; topic++)
			{
				IOUtil.println(topWordsByFreq(topic, 10));
			}
			for (int topic=0; topic<param.numTopics; topic++)
			{
				IOUtil.println(topPathsByFreq(topic, 20));
			}
		}
	}
	
	protected void sampleDoc(int doc)
	{
		int oldTopic,newTopic,sampleValue,word,numPaths,interval=getSampleInterval();
		TLDATopicNode newNode;
		for (int token=0; token<corpus.get(doc).docLength(); token+=interval)
		{			
			oldTopic=unassignTopicAndNode(doc, token);
			
			sampleValue=sampleTopicAndNode(doc, token, oldTopic);
			word=corpus.get(doc).getWord(token);
			numPaths=param.getNumPaths(word);
			newTopic=sampleValue/numPaths;
			newNode=topicNodeMaps.get(newTopic).get(word).get(sampleValue%numPaths);
			
			assignTopicAndNode(doc, token, newTopic, newNode);
		}
	}
	
	protected int unassignTopicAndNode(int doc, int token)
	{
		int oldTopic=corpus.get(doc).getTopicAssign(token);
		corpus.get(doc).unassignTopicAndNode(token);
		return oldTopic;
	}
	
	protected int sampleTopicAndNode(int doc, int token, int oldTopic)
	{
		double topicPathScores[]=computeTopicAndNodeScore(doc, token);
		int sampleValue=MathUtil.selectLogDiscrete(topicPathScores);
		if (sampleValue==-1)
		{
			sampleValue=randoms.nextInt(topicPathScores.length);
			IOUtil.println(format(topicPathScores));
		}
		
		return sampleValue;
	}
	
	protected void assignTopicAndNode(int doc, int token, int newTopic, TLDATopicNode newNode)
	{
		corpus.get(doc).assignTopicAndNode(token, newTopic, newNode);
	}
	
	protected double[] computeTopicAndNodeScore(int doc, int token)
	{
		int word=corpus.get(doc).getWord(token);
		int numPaths=param.getNumPaths(word);
		double scores[]=new double[param.numTopics*numPaths];
		for (int topic=0; topic<param.numTopics; topic++)
		{
			for (int path=0; path<numPaths; path++)
			{
				int idx=topic*numPaths+path;
				scores[idx]=Math.log(alpha[topic]+corpus.get(doc).getTopicCount(topic));
				if (type==TRAIN)
				{
					scores[idx]+=topicNodeMaps.get(topic).get(word).get(path).computePathLogProb(param.beta);
				}
				else
				{
					scores[idx]+=topicNodeMaps.get(topic).get(word).get(path).getPathLogProb();
				}
			}
		}
		return scores;
	}
	
	protected double[] backupComputeTopicAndNodeScore(int doc, int token)
	{
		int word=corpus.get(doc).getWord(token);
		int numPaths=param.getNumPaths(word);
		double scores[]=new double[param.numTopics*numPaths];
		for (int topic=0; topic<param.numTopics; topic++)
		{
			for (int path=0; path<numPaths; path++)
			{
				int idx=topic*numPaths+path;
				scores[idx]=Math.log(alpha[topic]+corpus.get(doc).getTopicCount(topic));
				if (type==TRAIN)
				{
					scores[idx]+=topicNodeMaps.get(topic).get(word).get(path).computePathLogProb(param.beta);
				}
				else
				{
					scores[idx]+=topicNodeMaps.get(topic).get(word).get(path).getPathLogProb();
				}
			}
		}
		return scores;
	}
	
	protected void updateHyperParam()
	{
		double oldAlpha[]=new double[param.numTopics];
		for (int topic=0; topic<param.numTopics; topic++)
		{
			oldAlpha[topic]=alpha[topic];
		}
		
		double numer;
		for (int topic=0; topic<param.numTopics; topic++)
		{
			numer=0.0;
			for (TLDADoc doc : corpus)
			{
				numer+=(double)(doc.getTopicCount(topic))/(double)(doc.getTopicCount(topic)+oldAlpha[topic]);
			}
			alpha[topic]=oldAlpha[topic]*numer/updateDenom;
		}
		
		double newAlphaSum=0.0;
		for (int topic=0; topic<param.numTopics; topic++)
		{
			newAlphaSum+=alpha[topic];
		}
		for (int topic=0; topic<param.numTopics; topic++)
		{
			alpha[topic]*=param.alpha*param.numTopics/newAlphaSum;
		}
	}
	
	public void readTheta(String thetaFileName) throws IOException
	{
		BufferedReader br=new BufferedReader(new InputStreamReader(new FileInputStream(thetaFileName), "UTF-8"));
		String line,seg[];
		for (int doc=0; doc<numDocs; doc++)
		{
			line=br.readLine();
			seg=line.split(" ");
			for (int topic=0; topic<param.numTopics; topic++)
			{
				theta[doc][topic]=(Double.valueOf(seg[topic])+param.alpha)/
						(corpus.get(doc).docLength()+param.alpha*param.numTopics);
			}
		}
		br.close();
	}
	
	public void computeLogLikelihood()
	{
		computeTheta();
		if (type==TRAIN)
		{
			computePathDist();
		}
		
		int word;
		double sum;
		logLikelihood=0.0;
		for (int doc=0; doc<numDocs; doc++)
		{
			int startPos=getStartPos();
			int interval=getSampleInterval();
			for (int token=startPos; token<corpus.get(doc).docLength(); token+=interval)
			{
				word=corpus.get(doc).getWord(token);
				sum=0.0;
				for (int topic=0; topic<param.numTopics; topic++)
				{
					int numPaths=param.getNumPaths(word);
					double pathProbSum=topicNodeMaps.get(topic).get(word).get(0).getPathLogProb();
					for (int path=1; path<numPaths; path++)
					{
						pathProbSum=MathUtil.logSum(pathProbSum,
								topicNodeMaps.get(topic).get(word).get(path).getPathLogProb());
					}
					sum+=theta[doc][topic]*Math.exp(pathProbSum);
				}
				logLikelihood+=Math.log(sum);
			}
		}
	}
	
	protected void computeTheta()
	{
		for (int doc=0; doc<numDocs; doc++)
		{
			for (int topic=0; topic<param.numTopics; topic++)
			{
				theta[doc][topic]=(alpha[topic]+corpus.get(doc).getTopicCount(topic))/
						(param.alpha*param.numTopics+getSampleSize(corpus.get(doc).docLength()));
			}
		}
	}
	
	protected void computePathDist()
	{
		for (int topic=0; topic<param.numTopics; topic++)
		{
			topics[topic].computeChildrenDist(param.beta);
			topics[topic].computePathLogProb();
		}
	}
	
	protected void getNumTestWords()
	{
//		if (type==TRAIN)
//		{
//			numTestWords=numWords;
//		}
//		else
//		{
//			numTestWords=0;
//			for (TLDADoc doc : corpus)
//			{
//				numTestWords+=doc.docLength()/2;
//			}
//		}
		numTestWords=numWords;
	}
	
	protected int getStartPos()
	{
//		return (type==TRAIN ? 0 : 1);
		return 0;
	}
	
	protected int getSampleSize(int docLength)
	{
//		return (type==TRAIN ? docLength : (docLength+1)/2);
		return docLength;
	}
	
	protected int getSampleInterval()
	{
//		return (type==TRAIN ? 1 : 2);
		return 1;
	}
	
	public double[][] getDocTopicDist()
	{
		return theta.clone();
	}
	
	public int getNumDocs()
	{
		return numDocs;
	}
	
	public int getNumWords()
	{
		return numWords;
	}
	
	public TLDADoc getDoc(int doc)
	{
		return corpus.get(doc);
	}
	
	public TLDATopicNode getTopic(int topic)
	{
		return topics[topic];
	}
	
	public double getLogLikelihood()
	{
		return logLikelihood;
	}
	
	public double getPerplexity()
	{
		return perplexity;
	}
	
	public int[][] getDocTopicCounts()
	{
		int docTopicCounts[][]=new int[numDocs][param.numTopics];
		for (int doc=0; doc<numDocs; doc++)
		{
			for (int topic=0; topic<param.numTopics; topic++)
			{
				docTopicCounts[doc][topic]=corpus.get(doc).getTopicCount(topic);
			}
		}
		return docTopicCounts;
	}
	
	public int[][] getTokenTopicAssign()
	{
		int tokenTopicAssign[][]=new int[numDocs][];
		for (int doc=0; doc<numDocs; doc++)
		{
			tokenTopicAssign[doc]=new int[corpus.get(doc).docLength()];
			for (int token=0; token<corpus.get(doc).docLength(); token++)
			{
				tokenTopicAssign[doc][token]=corpus.get(doc).getTopicAssign(token);
			}
		}
		return tokenTopicAssign;
	}
	
	public int[][] getTokenPathAssign()
	{
		int tokenPathAssign[][]=new int[numDocs][];
		for (int doc=0; doc<numDocs; doc++)
		{
			tokenPathAssign[doc]=new int[corpus.get(doc).docLength()];
			for (int token=0; token<corpus.get(doc).docLength(); token++)
			{
				tokenPathAssign[doc][token]=corpus.get(doc).getNodeAssign(token).getLeafNodeNo();
			}
		}
		return tokenPathAssign;
	}
	
	public String topPathsByFreq(int topic, int numTopPaths)
	{
		StringBuilder result=new StringBuilder("Topic "+topic+":");
		ArrayList<TLDAWord> words=new ArrayList<TLDAWord>();
		for (int vocab=0; vocab<param.numVocab; vocab++)
		{
			int numPaths=param.getNumPaths(vocab);
			for (int path=0; path<numPaths; path++)
			{
				words.add(new TLDAWord(param.getNode(vocab, path).getWordPath()+path,
						topicNodeMaps.get(topic).get(vocab).get(path).getSampledCounts()));
			}
		}
		
		Collections.sort(words);
		for (int i=0; i<numTopPaths; i++)
		{
			result.append("   "+words.get(i));
		}
		return result.toString();
	}
	
	public String topPathsByWeight(int topic, int numTopPaths)
	{
		StringBuilder result=new StringBuilder("Topic "+topic+":");
		ArrayList<TLDAWord> words=new ArrayList<TLDAWord>();
		for (int vocab=0; vocab<param.numVocab; vocab++)
		{
			int numPaths=param.getNumPaths(vocab);
			for (int path=0; path<numPaths; path++)
			{
				words.add(new TLDAWord(param.getNode(vocab, path).getWordPath()+path,
						Math.exp(topicNodeMaps.get(topic).get(vocab).get(path).getPathLogProb())));
			}
		}
		
		Collections.sort(words);
		for (int i=0; i<numTopPaths; i++)
		{
			result.append("   "+words.get(i));
		}
		return result.toString();
	}
	
	public String topWordsByFreq(int topic, int numTopWords)
	{
		StringBuilder result=new StringBuilder("Topic "+topic+":");
		TLDAWord words[]=new TLDAWord[param.numVocab];
		for (int vocab=0; vocab<param.numVocab; vocab++)
		{
			int count=0,numPaths=param.getNumPaths(vocab);
			for (int path=0; path<numPaths; path++)
			{
				count+=topicNodeMaps.get(topic).get(vocab).get(path).getSampledCounts();
			}
			words[vocab]=new TLDAWord(param.vocabList.get(vocab), count);
		}
		
		Arrays.sort(words);
		for (int i=0; i<numTopWords; i++)
		{
			result.append("   "+words[i]);
		}
		return result.toString();
	}
	
	public String topWordsByWeight(int topic, int numTopWords)
	{
		StringBuilder result=new StringBuilder("Topic "+topic+":");
		TLDAWord words[]=new TLDAWord[param.numVocab];
		for (int vocab=0; vocab<param.numVocab; vocab++)
		{
			int numPaths=param.getNumPaths(vocab);
			double weight=topicNodeMaps.get(topic).get(vocab).get(0).getPathLogProb();
			for (int path=1; path<numPaths; path++)
			{
				weight=MathUtil.logSum(weight,
						topicNodeMaps.get(topic).get(vocab).get(path).getPathLogProb());
			}
			words[vocab]=new TLDAWord(param.vocabList.get(vocab), Math.exp(weight));
		}
		
		Arrays.sort(words);
		for (int i=0; i<numTopWords; i++)
		{
			result.append("   "+words[i]);
		}
		return result.toString();
	}
	
	public double[][] getMarginalTopicWordDist()
	{
		double temp[][]=new double[param.numTopics][param.numVocab];
		for (int topic=0; topic<param.numTopics; topic++)
		{
			for (int vocab=0; vocab<param.numVocab; vocab++)
			{
				int numPaths=param.getNumPaths(vocab);
				double weight=topicNodeMaps.get(topic).get(vocab).get(0).getPathLogProb();
				for (int path=1; path<numPaths; path++)
				{
					weight=MathUtil.logSum(weight,
							topicNodeMaps.get(topic).get(vocab).get(path).getPathLogProb());
				}
				temp[topic][vocab]=Math.exp(weight);
			}
		}
		return temp;
	}
	
	public String[] getTopWordsByWeight(int topic, int numTopWords)
	{
		TLDAWord words[]=new TLDAWord[param.numVocab];
		for (int vocab=0; vocab<param.numVocab; vocab++)
		{
			int numPaths=param.getNumPaths(vocab);
			double weight=topicNodeMaps.get(topic).get(vocab).get(0).getPathLogProb();
			for (int path=1; path<numPaths; path++)
			{
				weight=MathUtil.logSum(weight,
						topicNodeMaps.get(topic).get(vocab).get(path).getPathLogProb());
			}
			words[vocab]=new TLDAWord(param.vocabList.get(vocab), Math.exp(weight));
		}
		
		Arrays.sort(words);
		String[] result=new String[numTopWords];
		for (int i=0; i<numTopWords; i++)
		{
			result[i]=words[i].getWord();
		}
		return result;
	}
	
	public double[][] getWordWeights(String word)
	{
		if (!param.vocabMap.containsKey(word)) return null;
		double weights[][]=new double[param.numTopics][];
		for (int topic=0; topic<param.numTopics; topic++)
		{
			weights[topic]=getWordWeights(topic, word);
		}
		return weights;
	}
	
	public double[] getWordWeights(int topic, String word)
	{
		if (!param.vocabMap.containsKey(word)) return null;
		
		int wordId=param.vocabMap.get(word);
		int numPaths=topicNodeMaps.get(topic).get(wordId).size();
		double weights[]=new double[numPaths];
		for (int path=0; path<numPaths; path++)
		{
			weights[path]=topicNodeMaps.get(topic).get(wordId).get(path).getWeight();
		}
		return weights;
	}
	
	public void writeResult(String resultFileName, int numTopWords) throws IOException
	{
		BufferedWriter bw=new BufferedWriter(new FileWriter(resultFileName));
		for (int topic=0; topic<param.numTopics; topic++)
		{
			bw.write(topWordsByFreq(topic, numTopWords));
			bw.newLine();
		}
		bw.close();
	}
	
	public void writePathResult(String resultFileName, int numTopPaths) throws IOException
	{
		BufferedWriter bw=new BufferedWriter(new FileWriter(resultFileName));
		for (int topic=0; topic<param.numTopics; topic++)
		{
			bw.write(topPathsByFreq(topic, numTopPaths));
			bw.newLine();
		}
		bw.close();
	}
	
	public void writeDocTopicDist(String docTopicDistFileName) throws IOException
	{
		BufferedWriter bw=new BufferedWriter(new FileWriter(docTopicDistFileName));
		IOUtil.writeMatrix(bw, theta);
		bw.close();
	}
	
	public void writeDocTopicCounts(String topicCountFileName) throws IOException
	{
		BufferedWriter bw=new BufferedWriter(new FileWriter(topicCountFileName));
		IOUtil.writeMatrix(bw, getDocTopicCounts());
		bw.close();
	}
	
	public void writeDocPathCounts(String pathCountFileName) throws IOException
	{
		BufferedWriter bw=new BufferedWriter(new FileWriter(pathCountFileName));
		for (int doc=0; doc<numDocs; doc++)
		{
			for (int path : corpus.get(doc).getPathSet())
			{
				bw.write(path+":"+corpus.get(doc).getPathCount(path)+" ");
			}
			bw.newLine();
		}
		bw.close();
	}
	
	public void writeTokenTopicAssign(String topicAssignFileName) throws IOException
	{
		BufferedWriter bw=new BufferedWriter(new FileWriter(topicAssignFileName));
		IOUtil.writeMatrix(bw, getTokenTopicAssign());
		bw.close();
	}
	
	public void writeTokenPathAssign(String pathAssignFileName) throws IOException
	{
		BufferedWriter bw=new BufferedWriter(new FileWriter(pathAssignFileName));
		IOUtil.writeMatrix(bw, getTokenPathAssign());
		bw.close();
	}
	
	public void writeModel(String modelFileName) throws IOException
	{
		BufferedWriter bw=new BufferedWriter(new FileWriter(modelFileName));
		bw.write(gson.toJson(alpha));
		bw.newLine();
		for (int topic=0; topic<param.numTopics; topic++)
		{
			topics[topic].prettyPrint(bw);
		}
		bw.close();
	}
	
	protected void initVariables()
	{
		corpus=new ArrayList<TLDADoc>();
		topics=new TLDATopicNode[param.numTopics];
		alpha=new double[param.numTopics];
		for (int topic=0; topic<param.numTopics; topic++)
		{
			topics[topic]=new TLDATopicNode();
			topics[topic].copyTree(param.topicPrior);
		}
	}
	
	protected void loadModel(String modelFileName) throws IOException
	{
		BufferedReader br=new BufferedReader(new InputStreamReader(new FileInputStream(modelFileName), "UTF-8"));
		String line=br.readLine();
		alpha=gson.fromJson(line, double[].class);
		for (int topic=0; topic<param.numTopics; topic++)
		{
			topics[topic]=TLDATopicNode.fromPrettyPrint(br);
			topics[topic].computePathLogProb();
		}
		br.close();
	}
	
	protected static String format(double num)
	{
		return Fourmat.format(num);
	}
	
	protected static String format(double nums[])
	{
		return Fourmat.format(nums);
	}
	
	static
	{
		randoms=new Randoms();
		gson=new Gson();
	}
	
	public TLDA(TLDAParam parameters)
	{
		this.type=TRAIN;
		this.param=parameters;
		initVariables();
		
		for (int topic=0; topic<param.numTopics; topic++)
		{
			alpha[topic]=param.alpha;
		}
	}
	
	public TLDA(TLDA LDATrain, TLDAParam parameters)
	{
		this.type=TEST;
		this.param=parameters;
		initVariables();
		
		alpha=LDATrain.alpha.clone();
		topics=LDATrain.topics.clone();
		for (int topic=0; topic<param.numTopics; topic++)
		{
			topics[topic].computePathLogProb();
		}
	}
	
	public TLDA(String modelFileName, TLDAParam parameters) throws IOException
	{
		this.type=TEST;
		this.param=parameters;
		initVariables();
		loadModel(modelFileName);
	}
	
	//TODO use gson to output alpha etc.
}

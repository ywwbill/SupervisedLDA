package lda;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.FileInputStream;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;

import cc.mallet.util.Randoms;
import lda.util.LDADoc;
import lda.util.LDATopic;
import lda.util.LDAWord;
import util.MathUtil;
import util.format.Fourmat;
import util.IOUtil;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.annotations.Expose;

public class LDA
{	
	public static final int TRAIN=0;
	public static final int TEST=1;
	
	public final LDAParam param;
	
	protected static Randoms randoms;
	protected static Gson gson;
	
	@Expose protected double alpha[];
	protected double updateDenom;

	protected int numDocs;
	protected int numWords;
	protected int numTestWords;
	protected final int type;
	
	protected ArrayList<LDADoc> corpus;
	protected LDATopic topics[];
	
	protected double theta[][];
	@Expose protected double phi[][];
	
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
				corpus.add(new LDADoc(line, param.numTopics, param.numVocab));
			}
			else
			{
				corpus.add(new LDADoc(line, param.numTopics, param.vocabMap));
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
		initDocVariables();
		initTopicAssigns();
		if (param.verbose) printParam();
	}
	
	protected void initTopicAssigns()
	{
		for (LDADoc doc : corpus)
		{
			int interval=getSampleInterval();
			for (int token=0; token<doc.docLength(); token+=interval)
			{
				int topic=randoms.nextInt(param.numTopics);
				doc.assignTopic(token, topic);
				
				int word=doc.getWord(token);
				topics[topic].addVocab(word);
			}
		}
	}
	
	public void initialize(String topicAssignFileName) throws IOException
	{
		initDocVariables();
		initTopicAssigns(topicAssignFileName);
		if (param.verbose) printParam();
	}
	
	protected void initTopicAssigns(String topicAssignFileName) throws IOException
	{
		BufferedReader br=new BufferedReader(new InputStreamReader(new FileInputStream(topicAssignFileName), "UTF-8"));
		String line,seg[];
		for (int doc=0; doc<numDocs; doc++)
		{
			line=br.readLine();
			seg=line.split(" ");
			int interval=getSampleInterval();
			for (int token=0; token<corpus.get(doc).docLength(); token+=interval)
			{
				int topic=randoms.nextInt(param.numTopics);
				if (token<seg.length && seg[token].length()>0)
				{
					topic=Integer.valueOf(seg[token]);
				}
				corpus.get(doc).assignTopic(token, topic);
				
				int word=corpus.get(doc).getWord(token);
				topics[topic].addVocab(word);
			}
		}
		br.close();
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
		}
	}
	
	protected void sampleDoc(int doc)
	{
		int oldTopic,newTopic,interval=getSampleInterval();
		for (int token=0; token<corpus.get(doc).docLength(); token+=interval)
		{			
			oldTopic=unassignTopic(doc, token);
			newTopic=sampleTopic(doc, token, oldTopic);
			assignTopic(doc, token, newTopic);
		}
	}
	
	protected int unassignTopic(int doc, int token)
	{
		int oldTopic=corpus.get(doc).getTopicAssign(token);
		int word=corpus.get(doc).getWord(token);
		corpus.get(doc).unassignTopic(token);
		topics[oldTopic].removeVocab(word);
		return oldTopic;
	}
	
	protected int sampleTopic(int doc, int token, int oldTopic)
	{
		int word=corpus.get(doc).getWord(token);
		double topicScores[]=new double[param.numTopics];
		for (int topic=0; topic<param.numTopics; topic++)
		{
			topicScores[topic]=topicUpdating(doc, topic, word);
		}
		
		int newTopic=MathUtil.selectLogDiscrete(topicScores);
		if (newTopic==-1)
		{
			newTopic=oldTopic;
			IOUtil.println(format(topicScores));
		}
		
		return newTopic;
	}
	
	protected void assignTopic(int doc, int token, int newTopic)
	{
		int word=corpus.get(doc).getWord(token);
		corpus.get(doc).assignTopic(token, newTopic);
		topics[newTopic].addVocab(word);
	}
	
	protected double topicUpdating(int doc, int topic, int vocab)
	{
		if (type==TRAIN)
		{
			return Math.log((alpha[topic]+corpus.get(doc).getTopicCount(topic))*
					(param.beta+topics[topic].getVocabCount(vocab))/
					(param.beta*param.numVocab+topics[topic].getTotalTokens()));
		}
		return Math.log((alpha[topic]+corpus.get(doc).getTopicCount(topic))*phi[topic][vocab]);
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
			for (LDADoc doc : corpus)
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
	
	protected void computeLogLikelihood()
	{
		computeTheta();
		if (type==TRAIN)
		{
			computePhi();
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
					sum+=theta[doc][topic]*phi[topic][word];
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
	
	protected void computePhi()
	{
		for (int topic=0; topic<param.numTopics; topic++)
		{
			for (int vocab=0; vocab<param.numVocab; vocab++)
			{
				phi[topic][vocab]=(param.beta+topics[topic].getVocabCount(vocab))/
						(param.beta*param.numVocab+topics[topic].getTotalTokens());
			}
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
//			for (LDADoc doc : corpus)
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
	
	public double[][] getTopicVocabDist()
	{
		return phi.clone();
	}
	
	public int getNumDocs()
	{
		return numDocs;
	}
	
	public int getNumWords()
	{
		return numWords;
	}
	
	public LDADoc getDoc(int doc)
	{
		return corpus.get(doc);
	}
	
	public LDATopic getTopic(int topic)
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
	
	public String topWordsByFreq(int topic, int numTopWords)
	{
		String result="Topic "+topic+":";
		LDAWord words[]=new LDAWord[param.numVocab];
		for (int vocab=0; vocab<param.numVocab; vocab++)
		{
			words[vocab]=new LDAWord(param.vocabList.get(vocab), topics[topic].getVocabCount(vocab));
		}
		
		Arrays.sort(words);
		for (int i=0; i<numTopWords; i++)
		{
			result+="   "+words[i];
		}
		return result;
	}
	
	public String topWordsByWeight(int topic, int numTopWords)
	{
		String result="Topic "+topic+":";
		LDAWord words[]=new LDAWord[param.numVocab];
		for (int vocab=0; vocab<param.numVocab; vocab++)
		{
			words[vocab]=new LDAWord(param.vocabList.get(vocab), phi[topic][vocab]);
		}
		
		Arrays.sort(words);
		for (int i=0; i<numTopWords; i++)
		{
			result+="   "+words[i];
		}
		return result;
	}
	
	public String[] getTopWordsByWeight(int topic, int numTopWords)
	{
		LDAWord words[]=new LDAWord[param.numVocab];
		for (int vocab=0; vocab<param.numVocab; vocab++)
		{
			words[vocab]=new LDAWord(param.vocabList.get(vocab), phi[topic][vocab]);
		}
		
		Arrays.sort(words);
		String result[]=new String[numTopWords];
		for (int i=0; i<numTopWords; i++)
		{
			result[i]=words[i].getWord();
		}
		return result;
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
	
	public void writeTokenTopicAssign(String topicAssignFileName) throws IOException
	{
		BufferedWriter bw=new BufferedWriter(new FileWriter(topicAssignFileName));
		IOUtil.writeMatrix(bw, getTokenTopicAssign());
		bw.close();
	}
	
	public void writeModel(String modelFileName) throws IOException
	{
		BufferedWriter bw=new BufferedWriter(new FileWriter(modelFileName));
		bw.write(gson.toJson(this));
		bw.close();
	}
	
	protected void initVariables()
	{
		corpus=new ArrayList<LDADoc>();
		topics=new LDATopic[param.numTopics];
		alpha=new double[param.numTopics];
		phi=new double[param.numTopics][param.numVocab];
		for (int topic=0; topic<param.numTopics; topic++)
		{
			topics[topic]=new LDATopic(param.numVocab);
		}
	}
	
	protected void copyModel(LDA LDAModel)
	{
		alpha=LDAModel.alpha.clone();
		phi=LDAModel.phi.clone();
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
		gson=new GsonBuilder().excludeFieldsWithoutExposeAnnotation().create();
	}
	
	public LDA(LDAParam parameters)
	{
		this.type=TRAIN;
		this.param=parameters;
		initVariables();
		
		for (int topic=0; topic<param.numTopics; topic++)
		{
			alpha[topic]=param.alpha;
		}
	}
	
	public LDA(LDA LDATrain, LDAParam parameters)
	{
		this.type=TEST;
		this.param=parameters;
		initVariables();
		copyModel(LDATrain);
	}
	
	public LDA(String modelFileName, LDAParam parameters) throws IOException
	{
		LDA LDATrain=gson.fromJson(new InputStreamReader(new FileInputStream(modelFileName), "UTF-8"), this.getClass());
		this.type=TEST;
		this.param=parameters;
		initVariables();
		copyModel(LDATrain);
	}
}

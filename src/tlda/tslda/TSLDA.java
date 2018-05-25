package tlda.tslda;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.InputStreamReader;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;

import cc.mallet.optimize.LimitedMemoryBFGS;
import tlda.TLDA;
import tlda.TLDAParam;
import tlda.util.TLDATopicNode;
import tlda.util.TLDAWord;
import util.IOUtil;
import util.MathUtil;
import com.google.gson.annotations.Expose;

public class TSLDA extends TLDA
{
	@Expose protected double eta[]; //topic
	@Expose protected double tau[]; //word
	@Expose protected double rho[]; //path
	
	protected int numLabels;
	protected double labels[];
	protected double predLabels[];
	protected boolean labelStatuses[];
	
	protected double weight;
	protected double error;
	
	public void readCorpus(String corpusFileName) throws IOException
	{
		super.readCorpus(corpusFileName);
		labels=new double[numDocs];
		predLabels=new double[numDocs];
		labelStatuses=new boolean[numDocs];
		numLabels=0;
		Arrays.fill(labelStatuses, false);
	}
	
	public void readLabels(String labelFileName) throws IOException
	{
		BufferedReader br=new BufferedReader(new InputStreamReader(new FileInputStream(labelFileName), "UTF-8"));
		String line;
		for (int doc=0; doc<numDocs; doc++)
		{
			line=br.readLine();
			if (line==null) break;
			if (corpus.get(doc).docLength()==0) continue;
			if (line.length()>0)
			{
				labels[doc]=Double.valueOf(line);
				labelStatuses[doc]=true;
				numLabels++;
			}
		}
		br.close();
	}
	
	protected void printParam()
	{
		super.printParam();
		param.printSLDAParam("\t");
		IOUtil.println("\t#labels: "+numLabels);
		IOUtil.println("\t#topic features: "+(param.topicFeature? param.numTopics : "False"));
		IOUtil.println("\tword features: "+(param.wordFeature? param.numVocab : "False"));
		IOUtil.println("\tpath features: "+(param.pathFeature? param.numLeafNodes : "False"));
	}
	
	public void sample(int numIters)
	{
		for (int iteration=1; iteration<=numIters; iteration++)
		{
			for (int doc=0; doc<numDocs; doc++)
			{
				weight=computeWeight(doc);
				sampleDoc(doc);
			}
			computeLogLikelihood();
			perplexity=Math.exp(-logLikelihood/numTestWords);
			
			if (type==TRAIN)
			{
				optimize();
			}
			
			if (param.verbose)
			{
				IOUtil.print("<"+iteration+">"+"\tLog-LLD: "+format(logLikelihood)+"\tPPX: "+format(perplexity));
			}
			computeError();
			if (param.verbose && numLabels>0)
			{
				IOUtil.print("\tError: "+format(error));
			}
			
			if (param.verbose) IOUtil.println();
			
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
	
	protected int unassignTopicAndNode(int doc, int token)
	{
		int oldTopic=corpus.get(doc).getTopicAssign(token);
		int oldPath=corpus.get(doc).getNodeAssign(token).getLeafNodeNo();
		corpus.get(doc).unassignTopicAndNode(token);
		if (param.topicFeature) weight-=eta[oldTopic]/corpus.get(doc).docLength();
		if (param.pathFeature) weight-=rho[oldPath]/corpus.get(doc).docLength();
		return oldTopic;
	}
	
	protected void assignTopicAndNode(int doc, int token, int newTopic, TLDATopicNode newNode)
	{
		corpus.get(doc).assignTopicAndNode(token, newTopic, newNode);
		if (param.topicFeature) weight+=eta[newTopic]/corpus.get(doc).docLength();
		if (param.pathFeature) weight+=rho[newNode.getLeafNodeNo()]/corpus.get(doc).docLength();
	}
	
	protected double[] computeTopicAndNodeScore(int doc, int token)
	{
		double scores[]=super.computeTopicAndNodeScore(doc, token);
		if (type!=TRAIN || !labelStatuses[doc] || corpus.get(doc).docLength()==0) return scores;
		
		int word=corpus.get(doc).getWord(token);
		int numPaths=param.getNumPaths(word);
		for (int topic=0; topic<param.numTopics; topic++)
		{
			for (int path=0; path<numPaths; path++)
			{
				int idx=topic*numPaths+path;
				double tempWeight=weight;
				if (param.topicFeature) tempWeight+=eta[topic]/corpus.get(doc).docLength();
				if (param.pathFeature)
				{
					int pathNo=topicNodeMaps.get(topic).get(word).get(path).getLeafNodeNo();
					tempWeight+=rho[pathNo]/corpus.get(doc).docLength();
				}
				scores[idx]-=MathUtil.sqr(labels[doc]-tempWeight)/(2.0*MathUtil.sqr(param.sigma));
			}
		}
		return scores;
	}
	
	protected void optimize()
	{
		TSLDAFunction optimizable=new TSLDAFunction(this);
		LimitedMemoryBFGS lbfgs=new LimitedMemoryBFGS(optimizable);
		try
		{
			lbfgs.optimize();
		}
		catch (Exception e)
		{
			e.printStackTrace();
			return;
		}
		
		int prev=0;
		if (param.topicFeature)
		{
			for (int topic=0; topic<param.numTopics; topic++)
			{
				eta[topic]=optimizable.getParameter(topic);
			}
			prev+=param.numTopics;
		}
		if (param.wordFeature)
		{
			for (int vocab=0; vocab<param.numVocab; vocab++)
			{
				tau[vocab]=optimizable.getParameter(prev+vocab);
			}
			prev+=param.numVocab;
		}
		if (param.pathFeature)
		{
			for (int path=0; path<param.numLeafNodes; path++)
			{
				rho[path]=optimizable.getParameter(prev+path);
			}
			prev+=param.numLeafNodes;
		}
	}
	
	protected double computeWeight(int doc)
	{
		double weight=0.0;
		if (corpus.get(doc).docLength()==0) return weight;
		if (param.topicFeature)
		{
			for (int topic=0; topic<param.numTopics; topic++)
			{
				weight+=eta[topic]*corpus.get(doc).getTopicCount(topic)/corpus.get(doc).docLength();
			}
		}
		if (param.wordFeature)
		{
			for (int word : corpus.get(doc).getWordSet())
			{
				weight+=tau[word]*corpus.get(doc).getWordCount(word)/corpus.get(doc).docLength();
			}
		}
		if (param.pathFeature)
		{
			for (int path : corpus.get(doc).getPathSet())
			{
				weight+=rho[path]*corpus.get(doc).getPathCount(path)/corpus.get(doc).docLength();
			}
		}
		return weight;
	}
	
	protected void computeError()
	{
		error=0.0;
		if (numLabels==0) return;
		for (int doc=0; doc<numDocs; doc++)
		{
			if (!labelStatuses[doc] || corpus.get(doc).docLength()==0) continue;
			error+=MathUtil.sqr(labels[doc]-computeWeight(doc));
		}
		error=Math.sqrt(error/(double)numLabels);
	}
	
	protected void computePredLabels()
	{
		for (int doc=0; doc<numDocs; doc++)
		{
			predLabels[doc]=computeWeight(doc);
		}
	}
	
	protected void getNumTestWords()
	{
		numTestWords=numWords;
	}
	
	protected int getStartPos()
	{
		return 0;
	}
	
	protected int getSampleSize(int docLength)
	{
		return docLength;
	}
	
	protected int getSampleInterval()
	{
		return 1;
	}
	
	public double getTopicWeight(int topic)
	{
		return eta[topic];
	}
	
	public double[] getTopicWeights()
	{
		return eta.clone();
	}
	
	public double getWordWeight(int word)
	{
		return tau[word];
	}
	
	public double[] getWordWeights()
	{
		return tau.clone();
	}
	
	public double getPathWeight(int path)
	{
		return rho[path];
	}
	
	public double[] getPathWeights()
	{
		return rho.clone();
	}
	
	public double[] getPredictions()
	{
		computePredLabels();
		return predLabels.clone();
	}
	
	public boolean getLabelStatus(int doc)
	{
		return labelStatuses[doc];
	}
	
	public double getResponseLabel(int doc)
	{
		return labels[doc];
	}
	
	public double getError()
	{
		return error;
	}
	
	public void writePredLabels(String predLabelFileName) throws IOException
	{
		computePredLabels();
		BufferedWriter bw=new BufferedWriter(new FileWriter(predLabelFileName));
		IOUtil.writeVector(bw, predLabels);
		bw.close();
	}
	
	public void writeRegValues(String regFileName) throws IOException
	{
		BufferedWriter bw=new BufferedWriter(new FileWriter(regFileName));
		for (int doc=0; doc<numDocs; doc++)
		{
			double reg=computeWeight(doc);
			bw.write(reg+"");
			bw.newLine();
		}
		bw.close();
	}
	
	public void writeResult(String resultFileName, int numTopWords) throws IOException
	{
		BufferedWriter bw=new BufferedWriter(new FileWriter(resultFileName));
		for (int topic=0; topic<param.numTopics; topic++)
		{
			bw.write((param.topicFeature? format(eta[topic]) : "")+"\t"+topWordsByFreq(topic, numTopWords));
			bw.newLine();
		}
		if (param.wordFeature)
		{
			bw.newLine();
			TLDAWord[] words=new TLDAWord[param.numVocab];
			for (int vocab=0; vocab<param.numVocab; vocab++)
			{
				words[vocab]=new TLDAWord(param.vocabList.get(vocab), tau[vocab]);
			}
			Arrays.sort(words);
			for (int i=0; i<numTopWords; i++)
			{
				bw.write(format(words[i].getWeight())+"\t"+words[i].getWord());
				bw.newLine();
			}
			bw.newLine();
			for (int i=param.numVocab-1; i>=param.numVocab-numTopWords; i--)
			{
				bw.write(format(words[i].getWeight())+"\t"+words[i].getWord());
				bw.newLine();
			}
		}
		if (param.pathFeature)
		{
			bw.newLine();
			TLDAWord[] words=new TLDAWord[param.numLeafNodes];
			for (int vocab=0; vocab<param.numVocab; vocab++)
			{
				for (int path=0; path<param.getNumPaths(vocab); path++)
				{
					int no=param.getNode(vocab, path).getLeafNodeNo();
					words[no]=new TLDAWord(param.vocabList.get(vocab)+path, rho[no]);
				}
			}
			Arrays.sort(words);
			for (int i=0; i<numTopWords; i++)
			{
				bw.write(format(words[i].getWeight())+"\t"+words[i].getWord());
				bw.newLine();
			}
			bw.newLine();
			for (int i=param.numVocab-1; i>=param.numVocab-numTopWords; i--)
			{
				bw.write(format(words[i].getWeight())+"\t"+words[i].getWord());
				bw.newLine();
			}
		}
		bw.close();
	}
	
	public void writePathResult(String resultFileName, int numTopPaths) throws IOException
	{
		BufferedWriter bw=new BufferedWriter(new FileWriter(resultFileName));
		for (int topic=0; topic<param.numTopics; topic++)
		{
			bw.write((param.topicFeature? format(eta[topic]) : "")+topPathsByFreq(topic, numTopPaths));
			bw.newLine();
		}
		bw.close();
	}
	
	public void writeModel(String modelFileName) throws IOException
	{
		BufferedWriter bw=new BufferedWriter(new FileWriter(modelFileName));
		bw.write(gson.toJson(alpha));
		bw.newLine();
		if (param.topicFeature)
		{
			bw.write(gson.toJson(eta));
			bw.newLine();
		}
		if (param.wordFeature)
		{
			bw.write(gson.toJson(tau));
			bw.newLine();
		}
		if (param.pathFeature)
		{
			bw.write(gson.toJson(rho));
			bw.newLine();
		}
		for (int topic=0; topic<param.numTopics; topic++)
		{
			topics[topic].prettyPrint(bw);
		}
		bw.close();
	}
	
	protected void initVariables()
	{
		super.initVariables();
		if (param.topicFeature) eta=new double[param.numTopics];
		if (param.wordFeature) tau=new double[param.numVocab];
		if (param.pathFeature) rho=new double[param.numLeafNodes];
	}
	
	protected void loadModel(String modelFileName) throws IOException
	{
		BufferedReader br=new BufferedReader(new InputStreamReader(new FileInputStream(modelFileName), "UTF-8"));
		String line=br.readLine();
		alpha=gson.fromJson(line, double[].class);
		
		if (param.topicFeature)
		{
			line=br.readLine();
			eta=gson.fromJson(line, double[].class);
		}
		
		if (param.wordFeature)
		{
			line=br.readLine();
			tau=gson.fromJson(line, double[].class);
		}
		
		if (param.pathFeature)
		{
			line=br.readLine();
			rho=gson.fromJson(line, double[].class);
		}
		
		for (int topic=0; topic<param.numTopics; topic++)
		{
			topics[topic]=TLDATopicNode.fromPrettyPrint(br);
			topics[topic].computePathLogProb();
		}
		br.close();
	}
	
	public TSLDA(TLDAParam parameters)
	{
		super(parameters);
	}
	
	public TSLDA(TSLDA LDATrain, TLDAParam parameters)
	{
		super(LDATrain, parameters);
		if (param.topicFeature) eta=LDATrain.eta.clone();
		if (param.wordFeature) tau=LDATrain.tau.clone();
		if (param.pathFeature) rho=LDATrain.rho.clone();
	}
	
	public TSLDA(String modelFileName, TLDAParam parameters) throws IOException
	{
		super(modelFileName, parameters);
	}
}

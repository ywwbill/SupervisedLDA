package lda.slda;

import java.io.IOException;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.FileInputStream;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.Arrays;

import lda.LDA;
import lda.LDAParam;
import lda.util.LDAWord;
import util.MathUtil;
import util.format.Fourmat;
import util.IOUtil;

import com.google.gson.annotations.Expose;
import cc.mallet.optimize.LimitedMemoryBFGS;

public class SLDA extends LDA
{
	@Expose protected double eta[];
	@Expose protected double tau[];
	
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
			
			if (type==TRAIN)
			{
				optimize();
			}
			if (param.verbose)
			{
				IOUtil.print("<"+iteration+">"+"\tLog-LLD: "+format(logLikelihood)+
						"\tPPX: "+format(perplexity));
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
		}
	}
	
	protected void sampleDoc(int doc)
	{
		int oldTopic,newTopic,interval=getSampleInterval();
		weight=computeWeight(doc);
		for (int token=0; token<corpus.get(doc).docLength(); token+=interval)
		{			
			oldTopic=unassignTopic(doc, token);
			if (type==TRAIN && labelStatuses[doc])
			{
				weight-=eta[oldTopic]/corpus.get(doc).docLength();
			}
			
			newTopic=sampleTopic(doc, token, oldTopic);
			
			assignTopic(doc, token, newTopic);
			if (type==TRAIN && labelStatuses[doc])
			{
				weight+=eta[newTopic]/corpus.get(doc).docLength();
			}
		}
	}
	
	protected double topicUpdating(int doc, int topic, int vocab)
	{
		double score=0.0;
		if (type==TRAIN)
		{
			score=Math.log((alpha[topic]+corpus.get(doc).getTopicCount(topic))*
					(param.beta+topics[topic].getVocabCount(vocab))/
					(param.beta*param.numVocab+topics[topic].getTotalTokens()));
		}
		else
		{
			score=Math.log((alpha[topic]+corpus.get(doc).getTopicCount(topic))*phi[topic][vocab]);
		}
		
		if (type==TRAIN && labelStatuses[doc])
		{
			score+=-MathUtil.sqr(labels[doc]-weight-eta[topic]/corpus.get(doc).docLength())/
					(2.0*MathUtil.sqr(param.sigma));
		}
		
		return score;
	}
	
	protected double computeWeight(int doc)
	{
		double weight=0.0;
		if (corpus.get(doc).docLength()==0) return weight;
		for (int topic=0; topic<param.numTopics; topic++)
		{
			weight+=eta[topic]*corpus.get(doc).getTopicCount(topic)/corpus.get(doc).docLength();
		}
		for (int vocab : corpus.get(doc).getWordSet())
		{
			weight+=tau[vocab]*corpus.get(doc).getWordCount(vocab)/corpus.get(doc).docLength();
		}
		return weight;
	}
	
	protected void optimize()
	{
		SLDAFunction optimizable=new SLDAFunction(this);
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
		for (int topic=0; topic<param.numTopics; topic++)
		{
			eta[topic]=optimizable.getParameter(topic);
		}
		for (int vocab=0; vocab<param.numVocab; vocab++)
		{
			tau[vocab]=optimizable.getParameter(vocab+param.numTopics);
		}
	}
	
	protected void computeError()
	{
		error=0.0;
		if (numLabels==0) return;
		for (int doc=0; doc<numDocs; doc++)
		{
			if (!labelStatuses[doc]) continue;
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
	
	public double computeAccuracy(double threshold)
	{
		if (numLabels==0) return 0.0;
		computePredLabels();
		int correctCount=0;
		for (int doc=0; doc<numDocs; doc++)
		{
			if (labelStatuses[doc] && (labels[doc]-threshold)*(predLabels[doc]-threshold)>0.0)
			{
				correctCount++;
			}
		}
		return (double)correctCount/(double)numLabels;
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
			bw.write(Fourmat.format(eta[topic])+"\t"+topWordsByFreq(topic, numTopWords));
			bw.newLine();
		}
		bw.newLine();
		LDAWord[] words=new LDAWord[param.numVocab];
		for (int vocab=0; vocab<param.numVocab; vocab++)
		{
			words[vocab]=new LDAWord(param.vocabList.get(vocab), tau[vocab]);
		}
		Arrays.sort(words);
		for (int i=0; i<numTopWords; i++)
		{
			bw.write(Fourmat.format(words[i].getWeight())+"\t"+words[i].getWord());
			bw.newLine();
		}
		bw.newLine();
		for (int i=param.numVocab-1; i>=param.numVocab-numTopWords; i--)
		{
			bw.write(Fourmat.format(words[i].getWeight())+"\t"+words[i].getWord());
			bw.newLine();
		}
		bw.close();
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
	
	public double getLexWeight(int vocab)
	{
		return tau[vocab];
	}
	
	public double[] getLexWeights()
	{
		return tau.clone();
	}
	
	public boolean getLabelStatus(int doc)
	{
		return labelStatuses[doc];
	}
	
	public double getResponseLabel(int doc)
	{
		return labels[doc];
	}
	
	public double getPrediction(int doc)
	{
		predLabels[doc]=computeWeight(doc);
		return predLabels[doc];
	}
	
	public double[] getPredictions()
	{
		computePredLabels();
		return predLabels.clone();
	}
	
	public double getError()
	{
		return error;
	}
	
	protected void initVariables()
	{
		super.initVariables();
		eta=new double[param.numTopics];
		tau=new double[param.numVocab];
	}
	
	protected void copyModel(LDA LDAModel)
	{
		super.copyModel(LDAModel);
		eta=((SLDA)LDAModel).eta.clone();
		tau=((SLDA)LDAModel).tau.clone();
	}
	
	public SLDA(LDAParam parameters)
	{
		super(parameters);
	}
	
	public SLDA(SLDA LDATrain, LDAParam parameters)
	{
		super(LDATrain, parameters);
	}
	
	public SLDA(String modelFileName, LDAParam parameters) throws IOException
	{
		super(modelFileName, parameters);
	}
}

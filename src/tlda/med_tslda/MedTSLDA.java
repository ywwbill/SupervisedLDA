package tlda.med_tslda;

import java.io.IOException;

import cc.mallet.optimize.LimitedMemoryBFGS;
import tlda.TLDAParam;
import tlda.tslda.TSLDA;
import util.IOUtil;
import util.MathUtil;

public class MedTSLDA extends TSLDA
{
	protected double zeta[];
	protected double lambda[];
	
	public void readCorpus(String corpusFileName) throws IOException
	{
		super.readCorpus(corpusFileName);
		zeta=new double[numDocs];
		lambda=new double[numDocs];
		for (int doc=0; doc<numDocs; doc++)
		{
			zeta[doc]=0.0;
			lambda[doc]=1.0;
		}
	}
	
	protected void printParam()
	{
		super.printParam();
		param.printHingeParam("\t");
	}
	
	public void sample(int numIters)
	{
		for (int iteration=1; iteration<=numIters; iteration++)
		{
			if (type==TRAIN)
			{
				optimize();
			}
			
			for (int doc=0; doc<numDocs; doc++)
			{
				weight=computeWeight(doc);
				sampleDoc(doc);
				computeZeta(doc);
				sampleLambda(doc);
			}
			computeLogLikelihood();
			perplexity=Math.exp(-logLikelihood/numTestWords);
			
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
	
	protected double[] computeTopicAndNodeScore(int doc, int token)
	{
		double scores[]=super.backupComputeTopicAndNodeScore(doc, token);
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
				double term1=param.c*(param.c*param.eps-lambda[doc])*Math.abs(labels[doc]-tempWeight)/lambda[doc];
				double term2=param.c*param.c*(labels[doc]-tempWeight)*(labels[doc]-tempWeight)/(2.0*lambda[doc]);
				scores[idx]+=term1-term2;
			}
		}
		return scores;
	}
	
	protected void optimize()
	{
		MedTSLDAFunction optimizable=new MedTSLDAFunction(this);
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
	
	protected void sampleLambda(int doc)
	{
		if (!labelStatuses[doc] || corpus.get(doc).docLength()==0) return;
		double newValue=MathUtil.sampleIG(1.0/(param.c*Math.abs(zeta[doc])), 1.0);
		lambda[doc]=1.0/newValue;
	}
	
	protected void computeZeta(int doc)
	{
		if (!labelStatuses[doc] || corpus.get(doc).docLength()==0) return;
		zeta[doc]=Math.abs(labels[doc]-computeWeight(doc))-param.eps;
	}
	
	public MedTSLDA(TLDAParam parameters)
	{
		super(parameters);
	}
	
	public MedTSLDA(MedTSLDA LDATrain, TLDAParam parameters)
	{
		super(LDATrain, parameters);
	}
	
	public MedTSLDA(String modelFileName, TLDAParam parameters) throws IOException
	{
		super(modelFileName, parameters);
	}
}

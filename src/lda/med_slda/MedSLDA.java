package lda.med_slda;

import java.io.IOException;

import cc.mallet.optimize.LimitedMemoryBFGS;
import lda.LDAParam;
import lda.slda.SLDA;
import util.IOUtil;
import util.MathUtil;

public class MedSLDA extends SLDA
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
			double tempWeight=weight+eta[topic]/corpus.get(doc).docLength();
			double term1=param.c*(param.c+lambda[doc])*labels[doc]*tempWeight/lambda[doc];
			double term2=param.c*param.c*tempWeight*tempWeight/(2.0*lambda[doc]);
			score+=term1-term2;
		}
		
		return score;
	}
	
	protected void optimize()
	{
		MedSLDAFunction optimizable=new MedSLDAFunction(this);
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
			tau[vocab]=optimizable.getParameter(param.numTopics+vocab);
		}
	}
	
	protected void sampleLambda(int doc)
	{
		if (!labelStatuses[doc]) return;
		double newValue=MathUtil.sampleIG(1.0/(param.c*Math.abs(zeta[doc])), 1.0);
		lambda[doc]=1.0/newValue;
	}
	
	protected void computeZeta(int doc)
	{
		if (!labelStatuses[doc]) return;
		zeta[doc]=Math.abs(labels[doc]-computeWeight(doc))-param.eps;
	}
	
	public MedSLDA(LDAParam parameters)
	{
		super(parameters);
	}
	
	public MedSLDA(MedSLDA LDATrain, LDAParam parameters)
	{
		super(LDATrain, parameters);
	}
	
	public MedSLDA(String modelFileName, LDAParam parameters) throws IOException
	{
		super(modelFileName, parameters);
	}
}

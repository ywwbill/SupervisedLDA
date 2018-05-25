package lda.slda;

import java.util.Arrays;

import cc.mallet.optimize.Optimizable.ByGradientValue;
import util.MathUtil;

public class SLDAFunction implements ByGradientValue
{
	private double eta[],tau[];
	private double etaGrad[],tauGrad[];
	private int numTopics,numVocab;
	private double nu,sigma,mu;
	private SLDA slda;
	
	public SLDAFunction(SLDA SLDAInst)
	{
		this.slda=SLDAInst;
		this.numTopics=slda.param.numTopics;
		this.numVocab=slda.param.numVocab;
		this.nu=slda.param.nu;
		this.sigma=slda.param.sigma;
		this.mu=slda.param.mu;
		
		eta=new double[numTopics];
		etaGrad=new double[numTopics];
		for (int topic=0; topic<numTopics; topic++)
		{
			eta[topic]=slda.getTopicWeight(topic);
		}
		
		tau=new double[numVocab];
		tauGrad=new double[numVocab];
		for (int vocab=0; vocab<numVocab; vocab++)
		{
			tau[vocab]=slda.getLexWeight(vocab);
		}
	}
	
	public double getValue()
	{
		double value=0.0,weight,sigmaDenom=2.0*sigma*sigma,nuDenom=2.0*nu*nu;
		for (int doc=0; doc<slda.getNumDocs(); doc++)
		{
			if (!slda.getLabelStatus(doc) || slda.getDoc(doc).docLength()==0) continue;
			weight=computeWeight(doc);
			value-=MathUtil.sqr(slda.getResponseLabel(doc)-weight)/sigmaDenom;
		}
		for (int topic=0; topic<numTopics; topic++)
		{
			value-=(eta[topic]-mu)*(eta[topic]-mu)/nuDenom;
		}
		for (int vocab=0; vocab<numVocab; vocab++)
		{
			value-=(tau[vocab]-mu)*(tau[vocab]-mu)/nuDenom;
		}
		return value;
	}
	
	public void getValueGradient(double gradient[])
	{
		Arrays.fill(etaGrad, 0.0);
		Arrays.fill(tauGrad, 0.0);
		double sigmaSq=sigma*sigma,nuSq=nu*nu;
		for (int doc=0; doc<slda.getNumDocs(); doc++)
		{
			if (!slda.getLabelStatus(doc) || slda.getDoc(doc).docLength()==0) continue;
			double weight=computeWeight(doc);
			double commonTerm=slda.getResponseLabel(doc)-weight;
			for (int topic=0; topic<numTopics; topic++)
			{
				etaGrad[topic]+=commonTerm*slda.getDoc(doc).getTopicCount(topic)/
						slda.getDoc(doc).docLength()/sigmaSq;
			}
			for (int vocab : slda.getDoc(doc).getWordSet())
			{
				tauGrad[vocab]+=commonTerm*slda.getDoc(doc).getWordCount(vocab)/
						slda.getDoc(doc).docLength()/sigmaSq;
			}
		}
		for (int topic=0; topic<numTopics; topic++)
		{
			gradient[topic]=etaGrad[topic]-(eta[topic]-mu)/nuSq;
		}
		for (int vocab=0; vocab<numVocab; vocab++)
		{
			gradient[numTopics+vocab]=tauGrad[vocab]-(tau[vocab]-mu)/nuSq;
		}
	}
	
	private double computeWeight(int doc)
	{
		double weight=0.0;
		if (slda.getDoc(doc).docLength()==0) return weight;
		for (int topic=0; topic<numTopics; topic++)
		{
			weight+=eta[topic]*slda.getDoc(doc).getTopicCount(topic)/slda.getDoc(doc).docLength();
		}
		for (int vocab : slda.getDoc(doc).getWordSet())
		{
			weight+=tau[vocab]*slda.getDoc(doc).getWordCount(vocab)/slda.getDoc(doc).docLength();
		}
		return weight;
	}
	
	public int getNumParameters()
	{
		return numTopics+numVocab;
	}
	
	public double getParameter(int i)
	{
		if (i<numTopics) return eta[i];
		return tau[i-numTopics];
	}
	
	public void getParameters(double buffer[])
	{
		for (int topic=0; topic<numTopics; topic++)
		{
			buffer[topic]=eta[topic];
		}
		for (int vocab=0; vocab<numVocab; vocab++)
		{
			buffer[numTopics+vocab]=tau[vocab];
		}
	}
	
	public void setParameter(int i, double r)
	{
		if (i<numTopics)
		{
			eta[i]=r;
			return;
		}
		tau[i-numTopics]=r;
	}
	
	public void setParameters(double newParameters[])
	{
		for (int topic=0; topic<numTopics; topic++)
		{
			eta[topic]=newParameters[topic];
		}
		for (int vocab=0; vocab<numVocab; vocab++)
		{
			tau[vocab]=newParameters[numTopics+vocab];
		}
	}
}

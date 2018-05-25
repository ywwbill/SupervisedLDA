package lda.med_slda;

import java.util.Arrays;

import cc.mallet.optimize.Optimizable.ByGradientValue;
import util.MathUtil;

public class MedSLDAFunction implements ByGradientValue
{
	private double eta[],tau[];
	private double etaGrad[],tauGrad[];
	private MedSLDA slda;
	private int numTopics,numVocab;
	private double nu,mu;
	
	public MedSLDAFunction(MedSLDA sldaInst)
	{
		this.slda=sldaInst;
		this.numTopics=slda.param.numTopics;
		this.numVocab=slda.param.numVocab;
		this.nu=slda.param.nu;
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
		double value=0.0,weight,nuDenom=2.0*nu*nu;
		for (int doc=0; doc<slda.getNumDocs(); doc++)
		{
			if (!slda.getLabelStatus(doc)) continue;
			weight=computeWeight(doc);
			value-=(MathUtil.sqr(slda.param.c*(slda.getResponseLabel(doc)-weight))-
					2.0*slda.param.c*(slda.param.c*slda.param.eps-slda.lambda[doc])*Math.abs(slda.getResponseLabel(doc)-weight))/
					(2.0*slda.lambda[doc]);
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
		double nuSq=nu*nu;
		for (int doc=0; doc<slda.getNumDocs(); doc++)
		{
			if (!slda.getLabelStatus(doc)) continue;
			double weight=computeWeight(doc);
			if (weight==slda.getResponseLabel(doc)) continue;
			double commonTerm1=slda.param.c*slda.param.c*(slda.getResponseLabel(doc)-weight);
			double commonTerm2=slda.param.c*(slda.param.c*slda.param.eps-slda.lambda[doc]);
			double commonTerm=(slda.getResponseLabel(doc)>weight ? commonTerm1-commonTerm2 : commonTerm1+commonTerm2)/(slda.lambda[doc]*slda.getDoc(doc).docLength());
			
			for (int topic=0; topic<numTopics; topic++)
			{
				etaGrad[topic]+=commonTerm*slda.getDoc(doc).getTopicCount(topic);
			}
			for (int vocab : slda.getDoc(doc).getWordSet())
			{
				tauGrad[vocab]+=commonTerm*slda.getDoc(doc).getWordCount(vocab);
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

package tlda.tslda;

import java.util.Arrays;

import cc.mallet.optimize.Optimizable.ByGradientValue;
import tlda.TLDAParam;

public class TSLDAFunction implements ByGradientValue
{
	private double eta[],tau[],rho[];
	private double etaGrad[],tauGrad[],rhoGrad[];
	private TSLDA slda;
	private int numTopics,numVocab,numPaths;
	private double nu,sigma,mu;
	private boolean topicFeature,wordFeature,pathFeature;
	private int priorType;
	
	public TSLDAFunction(TSLDA sldaInst)
	{
		this.slda=sldaInst;
		
		this.numTopics=slda.param.numTopics;
		this.numVocab=slda.param.numVocab;
		this.numPaths=slda.param.numLeafNodes;
		
		this.topicFeature=slda.param.topicFeature;
		this.wordFeature=slda.param.wordFeature;
		this.pathFeature=slda.param.pathFeature;
		
		this.nu=slda.param.nu;
		this.sigma=slda.param.sigma;
		this.mu=slda.param.mu;
		
		this.priorType=slda.param.prior;
		
		if (topicFeature)
		{
			eta=new double[numTopics];
			etaGrad=new double[numTopics];
			for (int topic=0; topic<numTopics; topic++)
			{
				eta[topic]=slda.getTopicWeight(topic);
			}
		}

		if (wordFeature)
		{
			tau=new double[numVocab];
			tauGrad=new double[numVocab];
			for (int vocab=0; vocab<numVocab; vocab++)
			{
				tau[vocab]=slda.getWordWeight(vocab);
			}
		}

		if (pathFeature)
		{
			rho=new double[numPaths];
			rhoGrad=new double[numPaths];
			for (int path=0; path<numPaths; path++)
			{
				rho[path]=slda.getPathWeight(path);
			}
		}
	}
	
	public double getValue()
	{
		double value=0.0,weight,sigmaDenom=2.0*sigma*sigma,nuDenom=2.0*nu*nu;
		for (int doc=0; doc<slda.getNumDocs(); doc++)
		{
			if (!slda.getLabelStatus(doc) || slda.getDoc(doc).docLength()==0) continue;
			weight=computeWeight(doc);
			value-=(slda.getResponseLabel(doc)-weight)*(slda.getResponseLabel(doc)-weight)/sigmaDenom;
		}
		
		if (topicFeature)
		{
			for (int topic=0; topic<numTopics; topic++)
			{
				switch (priorType)
				{
				case TLDAParam.LAPLACE: value-=Math.abs(eta[topic])/nu; break;
				case TLDAParam.GAUSSIAN:
				default: value-=(eta[topic]-mu)*(eta[topic]-mu)/nuDenom; break;
				}
			}
		}
		if (wordFeature)
		{
			for (int vocab=0; vocab<numVocab; vocab++)
			{
				switch (priorType)
				{
				case TLDAParam.LAPLACE: value-=Math.abs(tau[vocab])/nu; break;
				case TLDAParam.GAUSSIAN:
				default: value-=(tau[vocab]-mu)*(tau[vocab]-mu)/nuDenom; break;
				}
			}
		}
		if (pathFeature)
		{
			for (int path=0; path<numPaths; path++)
			{
				switch (priorType)
				{
				case TLDAParam.LAPLACE: value-=Math.abs(rho[path])/nu; break;
				case TLDAParam.GAUSSIAN:
				default: value-=(rho[path]-mu)*(rho[path]-mu)/nuDenom; break;
				}
			}
		}
		
		return value;
	}
	
	public void getValueGradient(double gradient[])
	{
		if (topicFeature) Arrays.fill(etaGrad, 0.0);
		if (wordFeature) Arrays.fill(tauGrad, 0.0);
		if (pathFeature) Arrays.fill(rhoGrad, 0.0);
		double sigmaSq=sigma*sigma,nuSq=nu*nu;
		for (int doc=0; doc<slda.getNumDocs(); doc++)
		{
			if (!slda.getLabelStatus(doc) || slda.getDoc(doc).docLength()==0) continue;
			double weight=computeWeight(doc);
			double commonTerm=slda.getResponseLabel(doc)-weight;
			
			if (topicFeature)
			{
				for (int topic=0; topic<numTopics; topic++)
				{
					etaGrad[topic]+=commonTerm*slda.getDoc(doc).getTopicCount(topic)/
							slda.getDoc(doc).docLength()/sigmaSq;
				}
			}
			
			if (wordFeature)
			{
				for (int vocab : slda.getDoc(doc).getWordSet())
				{
					tauGrad[vocab]+=commonTerm*slda.getDoc(doc).getWordCount(vocab)/
							slda.getDoc(doc).docLength()/sigmaSq;
				}
			}
			
			if (pathFeature)
			{
				for (int path : slda.getDoc(doc).getPathSet())
				{
					rhoGrad[path]+=commonTerm*slda.getDoc(doc).getPathCount(path)/
							slda.getDoc(doc).docLength()/sigmaSq;
				}
			}
		}
		
		int prevLen=0;
		if (topicFeature)
		{
			for (int topic=0; topic<numTopics; topic++)
			{
				switch (priorType)
				{
				case TLDAParam.LAPLACE: gradient[prevLen+topic]=etaGrad[topic]-Math.signum(eta[topic])/nu; break;
				case TLDAParam.GAUSSIAN:
				default: gradient[prevLen+topic]=etaGrad[topic]-(eta[topic]-mu)/nuSq; break;
				}
			}
			prevLen+=numTopics;
		}
		
		if (wordFeature)
		{
			for (int vocab=0; vocab<numVocab; vocab++)
			{
				switch (priorType)
				{
				case TLDAParam.LAPLACE: gradient[prevLen+vocab]=
						tauGrad[vocab]-Math.signum(tau[vocab])/nu; break;
				case TLDAParam.GAUSSIAN:
				default: gradient[prevLen+vocab]=tauGrad[vocab]-(tau[vocab]-mu)/nuSq; break;
				}
			}
			prevLen+=numVocab;
		}
		
		if (pathFeature)
		{
			for (int path=0; path<numPaths; path++)
			{
				switch (priorType)
				{
				case TLDAParam.LAPLACE: gradient[prevLen+path]=rhoGrad[path]-Math.signum(rho[path])/nu; break;
				case TLDAParam.GAUSSIAN:
				default: gradient[prevLen+path]=rhoGrad[path]-(rho[path]-mu)/nuSq; break;
				}
			}
			prevLen+=numPaths;
		}
	}
	
	private double computeWeight(int doc)
	{
		double weight=0.0;
		if (slda.getDoc(doc).docLength()==0) return weight;
		if (topicFeature)
		{
			for (int topic=0; topic<numTopics; topic++)
			{
				weight+=eta[topic]*slda.getDoc(doc).getTopicCount(topic)/
						slda.getDoc(doc).docLength();
			}
		}
		if (wordFeature)
		{
			for (int vocab : slda.getDoc(doc).getWordSet())
			{
				weight+=tau[vocab]*slda.getDoc(doc).getWordCount(vocab)/
						slda.getDoc(doc).docLength();
			}
		}
		if (pathFeature)
		{
			for (int path : slda.getDoc(doc).getPathSet())
			{
				weight+=rho[path]*slda.getDoc(doc).getPathCount(path)/
						slda.getDoc(doc).docLength();
			}
		}
		return weight;
	}
	
	public int getNumParameters()
	{
		int len=0;
		if (topicFeature) len+=numTopics;
		if (wordFeature) len+=numVocab;
		if (pathFeature) len+=numPaths;
		return len;
	}
	
	public double getParameter(int i)
	{
		if (topicFeature)
		{
			if (i<numTopics) return eta[i];
			i-=numTopics;
		}
		
		if (wordFeature)
		{
			if (i<numVocab) return tau[i];
			i-=numVocab;
		}
		
		return rho[i];
	}
	
	public void getParameters(double buffer[])
	{
		int prevLen=0;
		if (topicFeature)
		{
			for (int topic=0; topic<numTopics; topic++)
			{
				buffer[topic]=eta[topic];
			}
			prevLen+=numTopics;
		}
		
		if (wordFeature)
		{
			for (int vocab=0; vocab<numVocab; vocab++)
			{
				buffer[prevLen+vocab]=tau[vocab];
			}
			prevLen+=numVocab;
		}
		
		if (pathFeature)
		{
			for (int path=0; path<numPaths; path++)
			{
				buffer[prevLen+path]=rho[path];
			}
			prevLen+=numPaths;
		}
	}
	
	public void setParameter(int i, double r)
	{
		if (topicFeature)
		{
			if (i<numTopics)
			{
				eta[i]=r;
				return;
			}
			i-=numTopics;
		}
		
		if (wordFeature)
		{
			if (i<numVocab)
			{
				tau[i]=r;
				return;
			}
			i-=numVocab;
		}
		
		rho[i]=r;
	}
	
	public void setParameters(double newParameters[])
	{
		int prevLen=0;
		if (topicFeature)
		{
			for (int topic=0; topic<numTopics; topic++)
			{
				eta[topic]=newParameters[topic];
			}
			prevLen+=numTopics;
		}
		
		if (wordFeature)
		{
			for (int vocab=0; vocab<numVocab; vocab++)
			{
				tau[vocab]=newParameters[prevLen+vocab];
			}
			prevLen+=numVocab;
		}
		
		if (pathFeature)
		{
			for (int path=0; path<numPaths; path++)
			{
				rho[path]=newParameters[prevLen+path];
			}
			prevLen+=numPaths;
		}
	}
}

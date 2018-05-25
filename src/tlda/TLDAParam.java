package tlda;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import util.IOUtil;
import util.format.Fourmat;
import tlda.util.TLDATopicPriorNode;

public class TLDAParam
{
	//for topic model
	public double alpha=0.01;
	public double beta=0.01;
	public int numTopics=10;
	public boolean verbose=true;
	
	public boolean updateAlpha=false;
	public int updateAlphaInterval=10;
	
	public ArrayList<String> vocabList;
	public HashMap<String, Integer> vocabMap;
	public int numVocab;
	public TLDATopicPriorNode topicPrior;
	public ArrayList<List<List<Integer>>> vocabPaths;
	public int numLeafNodes;
	
	//for hinge loss
	public double c=1.0;
	public double eps=0.1;
	
	//for slda
	public double nu=1.0;
	public double mu=0.0;
	public double sigma=1.0;
	public boolean topicFeature=true;
	public boolean wordFeature=true;
	public boolean pathFeature=true;
	
	public static final int GAUSSIAN=0;
	public static final int L1=1;
	public static final int L2=2;
	public static final int LAPLACE=3;
	public int prior=GAUSSIAN;
	
	public void printBasicParam(String prefix)
	{
		IOUtil.println(prefix+"alpha: "+Fourmat.format(alpha));
		IOUtil.println(prefix+"beta: "+Fourmat.format(beta));
		IOUtil.println(prefix+"#topics: "+numTopics);
		IOUtil.println(prefix+"#vocab: "+numVocab);
		IOUtil.println(prefix+"verbose: "+verbose);
		IOUtil.println(prefix+"update alpha: "+updateAlpha);
		if (updateAlpha) IOUtil.println(prefix+"update alpha interval: "+updateAlphaInterval);
	}
	
	public void printSLDAParam(String prefix)
	{
		IOUtil.println(prefix+"mu: "+Fourmat.format(mu));
		IOUtil.println(prefix+"nu: "+Fourmat.format(nu));
		IOUtil.println(prefix+"sigma: "+Fourmat.format(sigma));
	}
	
	public void printHingeParam(String prefix)
	{
		IOUtil.println(prefix+"c: "+Fourmat.format(c));
		IOUtil.println(prefix+"epsilon: "+Fourmat.format(eps));
	}
	
	public TLDAParam(int numVocab)
	{
		vocabList=new ArrayList<String>();
		vocabMap=new HashMap<String, Integer>();
		this.numVocab=numVocab;
		for (int vocab=0; vocab<numVocab; vocab++)
		{
			vocabList.add(vocab+"");
			vocabMap.put(vocab+"", vocabMap.size());
		}
	}
	
	public TLDAParam(String vocabFileName, String vocabPriorFileName) throws IOException
	{
		vocabList=new ArrayList<String>();
		vocabMap=new HashMap<String, Integer>();
		BufferedReader br=new BufferedReader(new InputStreamReader(new FileInputStream(vocabFileName), "UTF-8"));
		String line;
		while ((line=br.readLine())!=null)
		{
			if (vocabMap.containsKey(line)) continue;
			vocabMap.put(line, vocabMap.size());
			vocabList.add(line);
		}
		br.close();
		numVocab=vocabList.size();
		
		if (vocabPriorFileName.length()>0)
		{
			topicPrior=TLDATopicPriorNode.fromPrettyPrint(vocabPriorFileName);
		}
		else
		{
			topicPrior=new TLDATopicPriorNode();
			for (int vocab=0; vocab<numVocab; vocab++)
			{
				topicPrior.addChild(new TLDATopicPriorNode(new TLDATopicPriorNode(vocabList.get(vocab), vocab)));
			}
		}
		
		vocabPaths=new ArrayList<List<List<Integer>>>();
		for (int vocab=0; vocab<numVocab; vocab++)
		{
			vocabPaths.add(new ArrayList<List<Integer>>());
		}
		topicPrior.getVocabPathMap(vocabPaths);
		numLeafNodes=topicPrior.getNumLeafNodes();
	}
	
	public int getNumPaths(int vocab)
	{
		return vocabPaths.get(vocab).size();
	}
	
	public TLDATopicPriorNode getNode(int vocab, int path)
	{
		return topicPrior.getNode(vocabPaths.get(vocab).get(path));
	}
}

package lda.util;

import java.util.HashMap;
import java.util.Set;

public class LDADoc
{	
	private int tokens[];
	private int topicAssigns[];
	private HashMap<Integer, Integer> wordCount;
	
	private int topicCounts[];
	
	public LDADoc(int numTopics, int numVocab)
	{
		this("", numTopics, numVocab);
	}
	
	public LDADoc(String document, int numTopics, int numVocab)
	{
		wordCount=new HashMap<Integer, Integer>();
		topicCounts=new int[numTopics];
		
		String seg[]=document.split(" "),segseg[];
		int len=Integer.valueOf(seg[0]);
		tokens=new int[len];
		topicAssigns=new int[len];
		int tempLen=0;
		for (int i=1; i<seg.length; i++)
		{
			if (seg[i].length()==0) continue;
			segseg=seg[i].split(":");
			int word=Integer.valueOf(segseg[0]);
			int count=Integer.valueOf(segseg[1]);
			assert(word>=0 && word<numVocab);
			
			if (!wordCount.containsKey(word))
			{
				wordCount.put(word, 0);
			}
			wordCount.put(word, wordCount.get(word)+count);
				
			for (int j=0; j<count; j++)
			{
				tokens[tempLen+j]=word;
				topicAssigns[tempLen+j]=-1;
			}
			tempLen+=count;
		}
	}
	
	public LDADoc(String document, int numTopics, HashMap<String, Integer> vocabMap)
	{
		topicCounts=new int[numTopics];
		
		String seg[]=document.split(" ");
		int len=0;
		for (int i=0; i<seg.length; i++)
		{
			if (seg[i].length()>0) len++;
		}
		tokens=new int[len];
		topicAssigns=new int[len];
		int tempLen=0;
		for (int i=0; i<seg.length; i++)
		{
			if (seg[i].length()==0) continue;
			int word=vocabMap.get(seg[i]);
			
			if (!wordCount.containsKey(word))
			{
				wordCount.put(word, 0);
			}
			wordCount.put(word, wordCount.get(word)+1);
			
			tokens[tempLen]=word;
			topicAssigns[tempLen]=-1;
			tempLen++;
		}
	}
	
	public int docLength()
	{
		return tokens.length;
	}
	
	public int getTopicAssign(int pos)
	{
		return topicAssigns[pos];
	}
	
	public void assignTopic(int pos, int topic)
	{
		int oldTopic=getTopicAssign(pos);
		topicAssigns[pos]=topic;
		if (oldTopic==-1)
		{
			topicCounts[topic]++;
		}
	}
	
	public void unassignTopic(int pos)
	{
		int oldTopic=getTopicAssign(pos);
		topicAssigns[pos]=-1;
		if (oldTopic!=-1)
		{
			topicCounts[oldTopic]--;
		}
	}
	
	public int getWord(int pos)
	{
		return tokens[pos];
	}
	
	public int getTopicCount(int topic)
	{
		return topicCounts[topic];
	}
	
	public Set<Integer> getWordSet()
	{
		return wordCount.keySet();
	}
	
	public int getWordCount(int word)
	{
		return (wordCount.containsKey(word)? wordCount.get(word) : 0);
	}
	
	public boolean containsWord(int word)
	{
		return wordCount.containsKey(word);
	}
}

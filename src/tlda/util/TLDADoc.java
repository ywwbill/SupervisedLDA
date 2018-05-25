package tlda.util;

import java.util.HashMap;
import java.util.TreeMap;
import java.util.Set;

public class TLDADoc
{	
	private int tokens[];
	private int topicAssigns[];
	private TLDATopicNode nodeAssigns[];
	private HashMap<Integer, Integer> wordCount;
	
	private int topicCounts[];
	private TreeMap<Integer, Integer> pathCounts;
	
	public TLDADoc(int numTopics, int numVocab, int numPaths)
	{
		this("", numTopics, numVocab, numPaths);
	}
	
	public TLDADoc(String document, int numTopics, int numVocab, int numPaths)
	{
		wordCount=new HashMap<Integer, Integer>();
		topicCounts=new int[numTopics];
		pathCounts=new TreeMap<Integer, Integer>();
		
		String seg[]=document.split(" "),segseg[];
		int len=Integer.valueOf(seg[0]);
		tokens=new int[len];
		topicAssigns=new int[len];
		nodeAssigns=new TLDATopicNode[len];
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
				nodeAssigns[tempLen+j]=null;
			}
			tempLen+=count;
		}
	}
	
	public TLDADoc(String document, int numTopics, HashMap<String, Integer> vocabMap, int numPaths)
	{
		topicCounts=new int[numTopics];
		pathCounts=new TreeMap<Integer, Integer>();
		
		String seg[]=document.split(" ");
		int len=0;
		for (int i=0; i<seg.length; i++)
		{
			if (seg[i].length()>0) len++;
		}
		tokens=new int[len];
		topicAssigns=new int[len];
		nodeAssigns=new TLDATopicNode[len];
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
			nodeAssigns[tempLen]=null;
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
	
	public TLDATopicNode getNodeAssign(int pos)
	{
		return nodeAssigns[pos];
	}
	
	public void assignTopicAndNode(int pos, int topic, TLDATopicNode node)
	{
		int oldTopic=getTopicAssign(pos);
		TLDATopicNode oldNode=getNodeAssign(pos);
		nodeAssigns[pos]=node;
		nodeAssigns[pos].assignPath();
		topicAssigns[pos]=topic;
		if (oldTopic==-1)
		{
			topicCounts[topic]++;
		}
		if (oldNode==null)
		{
			int no=node.getLeafNodeNo();
			if (!pathCounts.containsKey(no))
			{
				pathCounts.put(no, 0);
			}
			pathCounts.put(no, pathCounts.get(no)+1);
		}
	}
	
	public void unassignTopicAndNode(int pos)
	{
		int oldTopic=getTopicAssign(pos);
		TLDATopicNode oldNode=getNodeAssign(pos);
		nodeAssigns[pos].unassignPath();
		nodeAssigns[pos]=null;
		topicAssigns[pos]=-1;
		if (oldTopic!=-1)
		{
			topicCounts[oldTopic]--;
		}
		if (oldNode!=null)
		{
			int no=oldNode.getLeafNodeNo();
			pathCounts.put(no, pathCounts.get(no)-1);
			if (pathCounts.get(no)==0)
			{
				pathCounts.remove(no);
			}
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
	
	public int getPathCount(int path)
	{
		return (pathCounts.containsKey(path)? pathCounts.get(path) : 0);
	}
	
	public Set<Integer> getPathSet()
	{
		return pathCounts.keySet();
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

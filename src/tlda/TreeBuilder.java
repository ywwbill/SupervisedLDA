package tlda;

import java.io.IOException;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.FileInputStream;
import java.util.List;
import java.util.Map;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.PriorityQueue;

import util.IOUtil;
import util.Word;

import tlda.util.TLDATopicPriorNode;

public class TreeBuilder
{
	public void buildTree(String vocabFileName, String embedFileName, String treeFileName, int treeType, int numTop) throws IOException
	{
		IOUtil.println("Reading vocabulary ...");
		ArrayList<String> vocabList=IOUtil.loadStringList(vocabFileName);
		HashMap<String, Integer> vocabMap=IOUtil.loadStringIntegerMap(vocabFileName);
		int numVocab=vocabList.size();
		IOUtil.println("Vocabulary size is "+numVocab+".");
		double[][] embeds=new double[numVocab][];
		
		IOUtil.println("Reading embeddings ...");
		ArrayList<String> matchedVocab=new ArrayList<String>();
		BufferedReader br=new BufferedReader(new InputStreamReader(new FileInputStream(embedFileName), "UTF-8"));
		String line=br.readLine();
		String[] seg=line.split(" ");
		int numEmbed=Integer.valueOf(seg[0]),numDims=Integer.valueOf(seg[1]);
		for (int i=0; i<numEmbed; i++)
		{
			if ((i+1)%10000==0) IOUtil.println((i+1)+"/"+numEmbed);
			line=br.readLine();
			seg=line.split(" ");
			if (!vocabMap.containsKey(seg[0])) continue;
			int vid=vocabMap.get(seg[0]);
			embeds[vid]=new double[numDims];
			for (int j=0; j<numDims; j++)
			{
				embeds[vid][j]=Double.valueOf(seg[j+1]);
			}
			matchedVocab.add(seg[0]);
		}
		br.close();
		int numMatched=matchedVocab.size();
		IOUtil.println(numMatched+" words have embeddings.");
		
		IOUtil.println("Computing scores ...");
		double[][] score=new double[numMatched][numMatched];
		for (int i=0; i<numMatched; i++)
		{
			if ((i+1)%100==0) IOUtil.println((i+1)+"/"+numMatched);
			int vi=vocabMap.get(matchedVocab.get(i));
			double sum=0.0;
			for (int j=0; j<numMatched; j++)
			{
				if (i==j) continue;
				int vj=vocabMap.get(matchedVocab.get(j));
				for (int k=0; k<numDims; k++)
				{
					score[i][j]+=embeds[vi][k]*embeds[vj][k];
				}
				score[i][j]=Math.exp(score[i][j]);
				sum+=score[i][j];
			}
			for (int j=0; j<numMatched; j++)
			{
				if (i==j) continue;
				score[i][j]/=sum;
			}
		}
		
		TLDATopicPriorNode root;
		switch (treeType)
		{
		case 2: root=hac(matchedVocab, vocabMap, score, 1.0/(double)numMatched); break;
		case 3: root=hacWithLeafDup(matchedVocab, vocabMap, score, 1.0/(double)numMatched); break;
		default: root=build2LevelTree(matchedVocab, vocabMap, score, numTop); break;
		}
		
		IOUtil.println("Adding "+(numVocab-numMatched)+" words that have no embeddings ...");
		for (int i=0; i<numVocab; i++)
		{
			if (embeds[i]==null)
			{
				root.addChild(new TLDATopicPriorNode(vocabList.get(i), i));
			}
		}
		
		IOUtil.println("Printing tree ...");
		printTree(treeFileName, root);
	}
	
	public TLDATopicPriorNode build2LevelTree(List<String> vocab, Map<String, Integer> vocabMap, double[][] score, int numTop) throws IOException
	{		
		IOUtil.println("Building a two-level tree ...");
		TLDATopicPriorNode root=new TLDATopicPriorNode();
		int numVocab=vocab.size();
		ArrayList<Word> words=new ArrayList<Word>();
		for (int i=0; i<numVocab; i++)
		{
			if ((i+1)%100==0) IOUtil.println((i+1)+"/"+numVocab);
			words.clear();
			TLDATopicPriorNode internalNode=new TLDATopicPriorNode(vocab.get(i), vocabMap.get(vocab.get(i)));
			internalNode.addChild(new TLDATopicPriorNode(vocab.get(i), vocabMap.get(vocab.get(i))));
			for (int j=0; j<numVocab; j++)
			{
				if (score[i][j]==0.0) continue;
				words.add(new Word(vocab.get(j), score[i][j]));
			}
			Collections.sort(words);
			for (int j=0; j<numTop && j<words.size(); j++)
			{
				TLDATopicPriorNode leafNode=new TLDATopicPriorNode(words.get(j).getWord(), vocabMap.get(words.get(j).getWord()));
				internalNode.addChild(leafNode);
			}
			root.addChild(internalNode);
		}
		return root;
	}
	
	public TLDATopicPriorNode hac(List<String> vocab, Map<String, Integer> vocabMap, double[][] score, double threshold) throws IOException
	{
		IOUtil.println("Building an HAC tree ...");
		int numVocab=vocab.size();
		IOUtil.println("Adding clusters ...");
		ArrayList<Cluster> clusters=new ArrayList<Cluster>();
		ArrayList<Boolean> available=new ArrayList<Boolean>();
		ArrayList<TLDATopicPriorNode> nodes=new ArrayList<TLDATopicPriorNode>();
		for (int i=0; i<numVocab; i++)
		{
			if ((i+1)%1000==0) IOUtil.println((i+1)+"/"+numVocab);
			Cluster cluster=new Cluster(i);
			cluster.addWord(i);
			clusters.add(cluster);
			available.add(true);
			
			nodes.add(new TLDATopicPriorNode(vocab.get(i), vocabMap.get(vocab.get(i))));
		}
		int numAvailable=numVocab;
		
		IOUtil.println("Adding edges ...");
		PriorityQueue<Edge> edges=new PriorityQueue<Edge>();
		for (int i=0; i<numVocab; i++)
		{
			if ((i+1)%1000==0) IOUtil.println((i+1)+"/"+numVocab);
			for (int j=0; j<numVocab; j++)
			{
				if (i==j) continue;
				edges.add(new Edge(i, j, score[i][j]));
			}
		}
		
		IOUtil.println("Clustering ...");
		while (numAvailable>1)
		{
			if (numAvailable%100==0) IOUtil.println("numAvailable="+numAvailable+
					" #edges="+edges.size()+" topEdgeWeight="+edges.peek().weight);
			while ((!available.get(edges.peek().c1) || !available.get(edges.peek().c2)) && edges.peek().weight>=threshold)
			{
				edges.poll();
			}
			if (edges.peek().weight>=threshold)
			{
				Edge edge=edges.poll();
				Cluster newCluster=new Cluster(clusters.size());
				newCluster.addAllWord(clusters.get(edge.c1));
				newCluster.addAllWord(clusters.get(edge.c2));
				TLDATopicPriorNode newNode=new TLDATopicPriorNode();
				newNode.addChild(nodes.get(edge.c1));
				newNode.addChild(nodes.get(edge.c2));
				nodes.add(newNode);
				available.set(edge.c1, false);
				available.set(edge.c2, false);
				numAvailable-=2;
				for (int cid=0; cid<clusters.size(); cid++)
				{
					if (!available.get(cid)) continue;
					edges.add(new Edge(newCluster.ID, cid, newCluster.computeScore(clusters.get(cid), score)));
					edges.add(new Edge(cid, newCluster.ID, clusters.get(cid).computeScore(newCluster, score)));
				}
				clusters.add(newCluster);
				available.add(true);
				numAvailable++;
			}
			else
			{
				int c1=-1,c2=-1;
				for (int i=0; i<available.size(); i++)
				{
					if (!available.get(i)) continue;
					if (c1==-1)
					{
						c1=i;
					}
					else
					{
						if (c2==-1)
						{
							c2=i;
							break;
						}
					}
				}
				
				Cluster newCluster=new Cluster(clusters.size());
				newCluster.addAllWord(clusters.get(c1));
				newCluster.addAllWord(clusters.get(c2));
				TLDATopicPriorNode newNode=new TLDATopicPriorNode();
				newNode.addChild(nodes.get(c1));
				newNode.addChild(nodes.get(c2));
				nodes.add(newNode);
				available.set(c1, false);
				available.set(c2, false);
				numAvailable-=2;
				for (int cid=0; cid<clusters.size(); cid++)
				{
					if (!available.get(cid)) continue;
					edges.add(new Edge(newCluster.ID, cid, newCluster.computeScore(clusters.get(cid), score)));
					edges.add(new Edge(cid, newCluster.ID, clusters.get(cid).computeScore(newCluster, score)));
				}
				clusters.add(newCluster);
				available.add(true);
				numAvailable++;
			}
		}
		
		return nodes.get(nodes.size()-1);
	}
	
	public TLDATopicPriorNode hacWithLeafDup(List<String> vocab, Map<String, Integer> vocabMap, double[][] score, double threshold) throws IOException
	{
		IOUtil.println("Building an HAC tree with leaf duplication ...");
		int numVocab=vocab.size();
		IOUtil.println("Adding clusters ...");
		ArrayList<Cluster> clusters=new ArrayList<Cluster>();
		ArrayList<Boolean> available=new ArrayList<Boolean>();
		ArrayList<TLDATopicPriorNode> nodes=new ArrayList<TLDATopicPriorNode>();
		HashSet<Integer> addedPairs=new HashSet<Integer>();
		for (int i=0; i<numVocab; i++)
		{
			if ((i+1)%1000==0) IOUtil.println((i+1)+"/"+numVocab);
			double maxScore=0.0;
			int maxWord=-1;
			for (int j=0; j<numVocab; j++)
			{
				if (i==j || addedPairs.contains(j*numVocab+i)) continue;
				if (score[i][j]>maxScore)
				{
					maxScore=score[i][j];
					maxWord=j;
				}
			}
			
			if (maxWord==-1)
			{
				Cluster newCluster=new Cluster(i);
				newCluster.addWord(i);
				clusters.add(newCluster);
				available.add(true);
				
				nodes.add(new TLDATopicPriorNode(vocab.get(i), vocabMap.get(vocab.get(i))));
			}
			else
			{
				Cluster newCluster=new Cluster(i);
				newCluster.addWord(i);
				newCluster.addWord(maxWord);
				clusters.add(newCluster);
				available.add(true);
				
				TLDATopicPriorNode newNode=new TLDATopicPriorNode();
				newNode.addChild(new TLDATopicPriorNode(vocab.get(i), vocabMap.get(vocab.get(i))));
				newNode.addChild(new TLDATopicPriorNode(vocab.get(maxWord), vocabMap.get(vocab.get(maxWord))));
				nodes.add(newNode);
				
				addedPairs.add(i*numVocab+maxWord);
			}
		}
		int numAvailable=numVocab;
		
		IOUtil.println("Adding edges ...");
		PriorityQueue<Edge> edges=new PriorityQueue<Edge>();
		for (int i=0; i<numVocab; i++)
		{
			for (int j=0; j<numVocab; j++)
			{
				if (i==j) continue;
				edges.add(new Edge(i, j, clusters.get(i).computeScore(clusters.get(j), score)));
			}
		}
		
		IOUtil.println("Clustering ...");
		while (numAvailable>1)
		{
			if (numAvailable%100==0) IOUtil.println("numAvailable="+numAvailable+
					" #edges="+edges.size()+" topEdgeWeight="+edges.peek().weight);
			while ((!available.get(edges.peek().c1) || !available.get(edges.peek().c2)) && edges.peek().weight>=threshold)
			{
				edges.poll();
			}
			if (edges.peek().weight>=threshold)
			{
				Edge edge=edges.poll();
				Cluster newCluster=new Cluster(clusters.size());
				newCluster.addAllWord(clusters.get(edge.c1));
				newCluster.addAllWord(clusters.get(edge.c2));
				TLDATopicPriorNode newNode=new TLDATopicPriorNode();
				newNode.addChild(nodes.get(edge.c1));
				newNode.addChild(nodes.get(edge.c2));
				nodes.add(newNode);
				available.set(edge.c1, false);
				available.set(edge.c2, false);
				numAvailable-=2;
				for (int cid=0; cid<clusters.size(); cid++)
				{
					if (!available.get(cid)) continue;
					edges.add(new Edge(newCluster.ID, cid, newCluster.computeScore(clusters.get(cid), score)));
					edges.add(new Edge(cid, newCluster.ID, clusters.get(cid).computeScore(newCluster, score)));
				}
				clusters.add(newCluster);
				available.add(true);
				numAvailable++;
			}
			else
			{
				int c1=-1,c2=-1;
				for (int i=0; i<available.size(); i++)
				{
					if (!available.get(i)) continue;
					if (c1==-1)
					{
						c1=i;
					}
					else
					{
						if (c2==-1)
						{
							c2=i;
							break;
						}
					}
				}
				
				Cluster newCluster=new Cluster(clusters.size());
				newCluster.addAllWord(clusters.get(c1));
				newCluster.addAllWord(clusters.get(c2));
				TLDATopicPriorNode newNode=new TLDATopicPriorNode();
				newNode.addChild(nodes.get(c1));
				newNode.addChild(nodes.get(c2));
				nodes.add(newNode);
				available.set(c1, false);
				available.set(c2, false);
				numAvailable-=2;
				for (int cid=0; cid<clusters.size(); cid++)
				{
					if (!available.get(cid)) continue;
					edges.add(new Edge(newCluster.ID, cid, newCluster.computeScore(clusters.get(cid), score)));
					edges.add(new Edge(cid, newCluster.ID, clusters.get(cid).computeScore(newCluster, score)));
				}
				clusters.add(newCluster);
				available.add(true);
				numAvailable++;
			}
		}
		
		return nodes.get(nodes.size()-1);
	}
	
	public void printTree(String printTreeFileName,	TLDATopicPriorNode root) throws IOException
	{
		root.prettyPrint(printTreeFileName);
	}
	
	class Cluster
	{
		public HashSet<Integer> wordSet;
		public final int ID;
		
		public Cluster(int id)
		{
			this.ID=id;
			wordSet=new HashSet<Integer>();
		}
		
		public void addWord(int word)
		{
			wordSet.add(word);
		}
		
		public void addAllWord(Cluster c)
		{
			wordSet.addAll(c.wordSet);
		}
		
		public double computeScore(Cluster c, double scores[][])
		{
			double score=0.0;
			int num=0;
			for (int w1 : wordSet)
			{
				for (int w2 : c.wordSet)
				{
					if (w1!=w2)
					{
						score+=scores[w1][w2];
						num++;
					}
				}
			}
			return score/num;
		}
	}
	
	class Edge implements Comparable<Edge>
	{
		public int c1,c2;
		public double weight;
		
		public Edge(int c1, int c2, double weight)
		{
			this.c1=c1;
			this.c2=c2;
			this.weight=weight;
		}
		
		public boolean equals(Edge e)
		{
			return (this.c1==e.c1 && this.c2==e.c2);
		}
		
		public int compareTo(Edge e)
		{
			return -Double.compare(this.weight, e.weight);
		}
	}
}

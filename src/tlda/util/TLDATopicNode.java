package tlda.util;

import java.util.List;
import java.util.Stack;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.InputStreamReader;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedList;

import util.IOUtil;

public class TLDATopicNode
{
	private ArrayList<TLDATopicNode> children;
	private double weight;
	private double pathLogProb;
	private int sampledCounts;
	private TLDATopicNode father;
	private int leafNodeNo;
	
	public TLDATopicNode()
	{
		children=new ArrayList<TLDATopicNode>();
		weight=0.0;
		pathLogProb=0.0;
		sampledCounts=0;
		father=null;
		leafNodeNo=-1;
	}
	
	public void copyTree(TLDATopicPriorNode priorRoot)
	{
		LinkedList<TLDATopicNode> queue=new LinkedList<TLDATopicNode>();
		queue.add(this);
		LinkedList<TLDATopicPriorNode> priorQueue=new LinkedList<TLDATopicPriorNode>();
		priorQueue.add(priorRoot);
		
		TLDATopicPriorNode tempPrior=null;
		TLDATopicNode temp=null;
		while (!queue.isEmpty() && !priorQueue.isEmpty())
		{
			temp=queue.poll();
			tempPrior=priorQueue.poll();
			temp.leafNodeNo=tempPrior.getLeafNodeNo();
			for (int i=0; i<tempPrior.getNumChildren(); i++)
			{
				TLDATopicNode child=new TLDATopicNode();
				temp.addChild(child);
				queue.add(child);
				priorQueue.add(tempPrior.getChild(i));
			}
		}
	}
	
	public static TLDATopicNode fromPrettyPrint(String treeFileName) throws IOException
	{
		BufferedReader br=new BufferedReader(new InputStreamReader(new FileInputStream(treeFileName), "UTF-8"));
		TLDATopicNode root=fromPrettyPrint(br);
		br.close();
		return root;
	}
	
	public static TLDATopicNode fromPrettyPrint(BufferedReader br) throws IOException
	{
		Stack<TLDATopicNode> stack=new Stack<TLDATopicNode>();
		Stack<Integer> level=new Stack<Integer>();
		String line;
		TLDATopicNode root=null;
		while ((line=br.readLine())!=null)
		{
			if (line.equals("##")) break;
			int lv=line.indexOf('-');
			while (!stack.isEmpty() && level.peek()>=lv)
			{
				stack.pop();
				level.pop();
			}
			TLDATopicNode newNode=new TLDATopicNode();
			newNode.weight=Double.valueOf(line.substring(lv+1));
			if (!stack.isEmpty())
			{
				stack.peek().addChild(newNode);
			}
			else
			{
				root=newNode;
			}
			stack.push(newNode);
			level.push(lv);
		}
		root.computeLeafNodeNo();
		return root;
	}
	
	public void prettyPrint(BufferedWriter bw) throws IOException
	{
		Stack<String> indent=new Stack<String>();
		Stack<Boolean> last=new Stack<Boolean>();
		Stack<Integer> cid=new Stack<Integer>();
		Stack<TLDATopicNode> stack=new Stack<TLDATopicNode>();
		indent.push("");
		last.push(true);
		cid.add(-1);
		stack.push(this);
		
		TLDATopicNode temp;
		while (!stack.isEmpty())
		{
			temp=stack.peek();
			if (cid.peek()==-1)
			{
				bw.write(indent.peek());
				if (last.peek())
				{
					bw.write("\\-");
				}
				else
				{
					bw.write("|-");
				}
				bw.write(temp.weight+"");
				bw.newLine();
			}
			
			cid.push(cid.pop()+1);
			if (cid.peek()>=temp.getNumChildren())
			{
				indent.pop();
				last.pop();
				cid.pop();
				stack.pop();
			}
			else
			{
				if (last.peek())
				{
					indent.push(indent.peek()+"  ");
				}
				else
				{
					indent.push(indent.peek()+"| ");
				}
				last.push(cid.peek()==temp.getNumChildren()-1);
				stack.push(temp.getChild(cid.peek()));
				cid.push(-1);
			}
		}
		bw.write("##");
		bw.newLine();
	}
	
	public void prettyPrint(String treeFileName) throws IOException
	{
		BufferedWriter bw=new BufferedWriter(new FileWriter(treeFileName));
		prettyPrint(bw);
		bw.close();
	}
	
	public void prettyPrint()
	{
		Stack<String> indent=new Stack<String>();
		Stack<Boolean> last=new Stack<Boolean>();
		Stack<Integer> cid=new Stack<Integer>();
		Stack<TLDATopicNode> stack=new Stack<TLDATopicNode>();
		indent.push("");
		last.push(true);
		cid.add(-1);
		stack.push(this);
		
		TLDATopicNode temp;
		while (!stack.isEmpty())
		{
			temp=stack.peek();
			if (cid.peek()==-1)
			{
				IOUtil.print(indent.peek());
				if (last.peek())
				{
					IOUtil.print("\\-");
				}
				else
				{
					IOUtil.print("|-");
				}
				IOUtil.println(temp.weight);
			}
			
			cid.push(cid.pop()+1);
			if (cid.peek()>=temp.getNumChildren())
			{
				indent.pop();
				last.pop();
				cid.pop();
				stack.pop();
			}
			else
			{
				if (last.peek())
				{
					indent.push(indent.peek()+"  ");
				}
				else
				{
					indent.push(indent.peek()+"| ");
				}
				last.push(cid.peek()==temp.getNumChildren()-1);
				stack.push(temp.getChild(cid.peek()));
				cid.push(-1);
			}
		}
		IOUtil.println("##");
	}
	
	private void computeLeafNodeNo()
	{
		Stack<Integer> cid=new Stack<Integer>();
		Stack<TLDATopicNode> stack=new Stack<TLDATopicNode>();
		cid.add(-1);
		stack.push(this);
		
		int numLeafNodes=0;
		TLDATopicNode temp;
		while (!stack.isEmpty())
		{
			temp=stack.peek();
			if (cid.peek()==-1)
			{
				if (temp.isLeaf())
				{
					temp.leafNodeNo=numLeafNodes;
					numLeafNodes++;
				}
				else
				{
					temp.leafNodeNo=-1;
				}
			}
			
			cid.push(cid.pop()+1);
			if (cid.peek()>=temp.getNumChildren())
			{
				cid.pop();
				stack.pop();
			}
			else
			{
				stack.push(temp.getChild(cid.peek()));
				cid.push(-1);
			}
		}
	}
	
	public void assignPath()
	{
		TLDATopicNode temp=this;
		while (temp!=null)
		{
			temp.sampledCounts++;
			temp=temp.father;
		}
	}
	
	public void unassignPath()
	{
		TLDATopicNode temp=this;
		while (temp!=null)
		{
			temp.sampledCounts--;
			temp=temp.father;
		}
	}
	
	public void computeChildrenDist(double beta)
	{
		LinkedList<TLDATopicNode> queue=new LinkedList<TLDATopicNode>();
		queue.add(this);
		TLDATopicNode temp;
		while (!queue.isEmpty())
		{
			temp=queue.poll();
			temp.weight=(temp.isRoot()? 1.0 : (temp.sampledCounts+beta)/
					(temp.father.sampledCounts+temp.father.getNumChildren()*beta));
			for (int i=0; i<temp.getNumChildren(); i++)
			{
				queue.add(temp.getChild(i));
			}
		}
	}
	
	public void computePathLogProb()
	{
		LinkedList<TLDATopicNode> queue=new LinkedList<TLDATopicNode>();
		queue.add(this);
		TLDATopicNode temp;
		while (!queue.isEmpty())
		{
			temp=queue.poll();
			temp.pathLogProb=(temp.isRoot()? 0.0 : temp.father.pathLogProb+Math.log(temp.weight));
			for (int i=0; i<temp.getNumChildren(); i++)
			{
				queue.add(temp.getChild(i));
			}
		}
	}
	
	public double computePathLogProb(double beta) // for leaf node in sampling
	{
		TLDATopicNode temp=this;
		double logProb=0.0;
		while (temp.father!=null)
		{
			logProb+=Math.log((temp.sampledCounts+beta)/
					(temp.father.sampledCounts+temp.father.getNumChildren()*beta));
			temp=temp.father;
		}
		return logProb;
	}
	
	public void addChild(TLDATopicNode child)
	{
		child.setFather(this);
		children.add(child);
	}
	
	public void setFather(TLDATopicNode newFather)
	{
		father=newFather;
	}
	
	public boolean isRoot()
	{
		return father==null;
	}
	
	public boolean isLeaf()
	{
		return children.size()==0;
	}
	
	public int getNumChildren()
	{
		return children.size();
	}
	
	public TLDATopicNode getChild(int no)
	{
		return (no>=0 && no<children.size()? children.get(no) : null);
	}
	
	public TLDATopicNode getFather()
	{
		return father;
	}
	
	public TLDATopicNode getNode(List<Integer> path)
	{
		TLDATopicNode temp=this;
		for (int i=0; i<path.size(); i++)
		{
			if (path.get(i)<0 || path.get(i)>=temp.children.size()) return null;
			temp=temp.getChild(path.get(i));
		}
		return temp;
	}
	
	public double getPathLogProb()
	{
		return pathLogProb;
	}
	
	public double getWeight()
	{
		return weight;
	}
	
	public int getSampledCounts()
	{
		return sampledCounts;
	}
	
	public int getLeafNodeNo()
	{
		return leafNodeNo;
	}
}

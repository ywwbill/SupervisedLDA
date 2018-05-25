package tlda.util;

import util.format.Fourmat;

public class TLDAWord implements Comparable<TLDAWord>
{
	private String word;
	private int count;
	private double weight;
	
	private boolean compareByCount;
	
	public String getWord()
	{
		return word;
	}
	
	public int getCount()
	{
		return count;
	}
	
	public double getWeight()
	{
		return weight;
	}
	
	public TLDAWord(String word, int count)
	{
		this.word=word;
		this.count=count;
		compareByCount=true;
	}
	
	public TLDAWord(String word, double weight)
	{
		this.word=word;
		this.weight=weight;
		compareByCount=false;
	}
	
	public String toString()
	{
		if (compareByCount) return word+":"+count;
		return word+":"+Fourmat.format(weight);
	}
	
	public int compareTo(TLDAWord o)
	{
		if (compareByCount) return -Integer.compare(this.count, o.count);
		return -Double.compare(this.weight, o.weight);
	}
	
	public boolean equals(Object o)
	{
		if (!(o instanceof TLDAWord)) return false;
		return word.equals(((TLDAWord)o).word);
	}
}

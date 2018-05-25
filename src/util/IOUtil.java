package util;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.FileInputStream;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.HashMap;

public class IOUtil
{
	public static void print(Object obj)
	{
		System.out.print(obj);
	}
	
	public static void println(Object obj)
	{
		System.out.println(obj);
	}
	
	public static void println()
	{
		System.out.println();
	}
	
	public void printMatrix(int matrix[][])
	{
		for (int i=0; i<matrix.length; i++)
		{
			for (int j=0; j<matrix[i].length; j++)
			{
				IOUtil.print(matrix[i][j]+" ");
			}
			IOUtil.println();
		}
	}
	
	public static void readMatrix(String fileName, double matrix[][]) throws IOException
	{
		BufferedReader br=new BufferedReader(new InputStreamReader(new FileInputStream(fileName), "UTF-8"));
		readMatrix(br, matrix);
		br.close();
	}
	
	public static void readMatrix(BufferedReader br, double matrix[][]) throws IOException
	{
		readMatrix(br, matrix, matrix.length, matrix[0].length);
	}
	
	public static void readMatrix(BufferedReader br, double matrix[][], int dim1, int dim2) throws IOException
	{
		String line,seg[];
		for (int i=0; i<dim1; i++)
		{
			line=br.readLine();
			seg=line.split(" ");
			for (int j=0; j<dim2; j++)
			{
				matrix[i][j]=Double.valueOf(seg[j]);
			}
		}
	}
	
	public static void readVector(BufferedReader br, double vector[]) throws IOException
	{
		readVector(br, vector, vector.length);
	}
	
	public static void readVector(BufferedReader br, double vector[], int dim) throws IOException
	{
		String line;
		for (int i=0; i<dim; i++)
		{
			line=br.readLine();
			vector[i]=Double.valueOf(line);
		}
	}
	
	public static void writeMatrix(BufferedWriter bw, int matrix[][]) throws IOException
	{
		for (int i=0; i<matrix.length; i++)
		{
			for (int j=0; j<matrix[i].length; j++)
			{
				bw.write(matrix[i][j]+" ");
			}
			bw.newLine();
		}
	}
	
	public static void writeMatrix(BufferedWriter bw,
			ArrayList<ArrayList<Double>> matrix) throws IOException
	{
		for (int i=0; i<matrix.size(); i++)
		{
			for (int j=0; j<matrix.get(i).size(); i++)
			{
				bw.write(matrix.get(i).get(j)+" ");
			}
			bw.newLine();
		}
	}
	
	public static void writeMatrix(BufferedWriter bw, double matrix[][]) throws IOException
	{
		for (int i=0; i<matrix.length; i++)
		{
			for (int j=0; j<matrix[i].length; j++)
			{
				bw.write(matrix[i][j]+" ");
			}
			bw.newLine();
		}
	}
	
	public static void writeVector(BufferedWriter bw, double vector[]) throws IOException
	{
		for (int i=0; i<vector.length; i++)
		{
			bw.write(vector[i]+"");
			bw.newLine();
		}
	}
	
	public static void writeVector(BufferedWriter bw, int vector[]) throws IOException
	{
		for (int i=0; i<vector.length; i++)
		{
			bw.write(vector[i]+"");
			bw.newLine();
		}
	}
	
	public static void copyFile(String srcFileName, String destFileName) throws IOException
	{
		BufferedReader br=new BufferedReader(new InputStreamReader(new FileInputStream(srcFileName), "UTF-8"));
		BufferedWriter bw=new BufferedWriter(new FileWriter(destFileName));
		String line;
		while ((line=br.readLine())!=null)
		{
			bw.write(line);
			bw.newLine();
		}
		br.close();
		bw.close();
	}
	
	public static void mergeFiles(String files[], String fileName) throws IOException
	{
		BufferedWriter bw=new BufferedWriter(new FileWriter(fileName));
		String line;
		for (int i=0; i<files.length; i++)
		{
			BufferedReader br=new BufferedReader(new InputStreamReader(new FileInputStream(files[i]), "UTF-8"));
			while ((line=br.readLine())!=null)
			{
				bw.write(line);
				bw.newLine();
			}
			br.close();
		}
		bw.close();
	}
	
	public static ArrayList<String> loadStringList(String fileName) throws IOException
	{
		BufferedReader br=new BufferedReader(new InputStreamReader(new FileInputStream(fileName), "UTF-8"));
		String line;
		ArrayList<String> list=new ArrayList<String>();
		while ((line=br.readLine())!=null)
		{
			list.add(line);
		}
		br.close();
		return list;
	}
	
	public static ArrayList<Integer> loadIntegerList(String fileName) throws IOException
	{
		BufferedReader br=new BufferedReader(new InputStreamReader(new FileInputStream(fileName), "UTF-8"));
		String line;
		ArrayList<Integer> list=new ArrayList<Integer>();
		while ((line=br.readLine())!=null)
		{
			list.add(Integer.valueOf(line));
		}
		br.close();
		return list;
	}
	
	public static ArrayList<Double> loadDoubleList(String fileName) throws IOException
	{
		BufferedReader br=new BufferedReader(new InputStreamReader(new FileInputStream(fileName), "UTF-8"));
		String line;
		ArrayList<Double> list=new ArrayList<Double>();
		while ((line=br.readLine())!=null)
		{
			list.add(Double.valueOf(line));
		}
		br.close();
		return list;
	}
	
	public static HashSet<String> loadStringSet(String fileName) throws IOException
	{
		BufferedReader br=new BufferedReader(new InputStreamReader(new FileInputStream(fileName), "UTF-8"));
		String line;
		HashSet<String> set=new HashSet<String>();
		while ((line=br.readLine())!=null)
		{
			set.add(line);
		}
		br.close();
		return set;
	}
	
	public static HashMap<String, Integer> loadStringIntegerMap(String fileName) throws IOException
	{
		BufferedReader br=new BufferedReader(new InputStreamReader(new FileInputStream(fileName), "UTF-8"));
		String line;
		HashMap<String, Integer> map=new HashMap<String, Integer>();
		while ((line=br.readLine())!=null)
		{
			map.put(line, map.size());
		}
		br.close();
		return map;
	}
}

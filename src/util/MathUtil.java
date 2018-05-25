package util;

import java.util.List;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;

import cc.mallet.util.Randoms;

public class MathUtil
{
	public static Randoms randoms=new Randoms();
	
	//sampling
	public static int selectDiscrete(double score[])
	{
		if (score.length==1) return 0;
		double sum=0.0;
		for (int i=0; i<score.length; i++)
		{
			sum+=score[i];
		}
		
		double sample=randoms.nextDouble()*sum;
		int index=-1;
		while (sample>0 && index<score.length-1)
		{
			index++;
			sample-=score[index];
		}
		
		return index;
	}
	
	public static int selectLogDiscrete(double score[])
	{
		double max=selectMax(score);
		for (int i=0; i<score.length; i++)
		{
			score[i]=Math.exp(score[i]-max);
		}
		return selectDiscrete(score);
	}
	
	//array and matrix math
	public static double selectMax(double score[])
	{
		double max=Double.NEGATIVE_INFINITY;
		for (int i=0; i<score.length; i++)
		{
			if (score[i]>max)
			{
				max=score[i];
			}
		}
		return max;
	}
	
	public static double average(double nums[])
	{
		double avg=0.0;
		for (int i=0; i<nums.length; i++)
		{
			avg+=nums[i];
		}
		return avg/(double)nums.length;
	}
	
	public static double average(ArrayList<Double> nums)
	{
		double avg=0.0;
		for (int i=0; i<nums.size(); i++)
		{
			avg+=nums.get(i);
		}
		return avg/(double)nums.size();
	}
	
	public static double sum(double nums[])
	{
		double result=0.0;
		for (int i=0; i<nums.length; i++)
		{
			result+=nums[i];
		}
		return result;
	}
	
	public static void normalize(double vector[])
	{
		double len=0.0;
		for (int i=0; i<vector.length; i++)
		{
			len+=sqr(vector[i]);
		}
		len=Math.sqrt(len);
		for (int i=0; i<vector.length; i++)
		{
			vector[i]/=len;
		}
	}
	
	public static double matrixAbsDiff(double m1[][], double m2[][])
	{
		if (m1.length!=m2.length) return Double.MAX_VALUE;
		int numElements=0;
		double diff=0.0;
		for (int i=0; i<m1.length; i++)
		{
			if (m1[i].length!=m2[i].length) return Double.MAX_VALUE;
			numElements+=m1[i].length;
			for (int j=0; j<m1[i].length; j++)
			{
				diff+=Math.abs(m1[i][j]-m2[i][j]);
			}
		}
		if (numElements==0) return 0.0;
		return diff/numElements;
	}
	
	public static double matrixKLDivergence(double m1[][], double m2[][])
	{
		if (m1.length!=m2.length) return Double.MAX_VALUE;
		double avgKL=0.0;
		for (int i=0; i<m1.length; i++)
		{
			if (m1[i].length!=m2[i].length) return Double.MAX_VALUE;
			avgKL+=vectorKLDivergence(m1[i], m2[i]);
		}
		return avgKL/m1.length;
	}
	
	public static double vectorAbsDiff(double v1[], double v2[])
	{
		if (v1.length!=v2.length) return Double.MAX_VALUE;
		double diff=0.0;
		for (int i=0; i<v1.length; i++)
		{
			diff+=Math.abs(v1[i]-v2[i]);
		}
		return diff/v1.length;
	}
	
	public static double vectorKLDivergence(double v1[], double v2[])
	{
		if (v1.length!=v2.length) return Double.MAX_VALUE;
		double kl=0.0;
		for (int i=0; i<v1.length; i++)
		{
			kl+=v1[i]*Math.log(v1[i]/v2[i]);
		}
		return kl;
	}
	
	public static double vectorEuclidDistance(double v1[], double v2[])
	{
		if (v1.length!=v2.length) return Double.MAX_VALUE;
		double dis=0.0;
		for (int i=0; i<v1.length; i++)
		{
			dis+=sqr(v1[i]-v2[i]);
		}
		return Math.sqrt(dis);
	}
	
	public static int getLength(double vec[])
	{
		if (vec==null) return 0;
		return vec.length;
	}
	
	public static int getLength(List<Double> vec)
	{
		if (vec==null) return 0;
		return vec.size();
	}
	
	//useful functions
	public static double logFactorial(int n)
	{
		double result=0.0;
		for (int i=1; i<=n; i++)
		{
			result+=Math.log((double)i);
		}
		return result;
	}
	
	public static double sigmoid(double x)
	{
		return 1.0/(1.0+Math.exp(-1.0*x));
	}
	
	public static double sqr(double x)
	{
		return x*x;
	}
	
	public static void computePRF(int predLabels[], int trueLabels[], int label)
	{
		int truePos=0,falsePos=0,falseNeg=0;
		for (int i=0; i<predLabels.length; i++)
		{
			if (predLabels[i]==label && trueLabels[i]==label)
			{
				truePos++;
			}
			if (predLabels[i]==label && trueLabels[i]!=label)
			{
				falsePos++;
			}
			if (predLabels[i]!=label && trueLabels[i]==label)
			{
				falseNeg++;
			}
		}
		double precision=(truePos+falsePos>0? (double)truePos/(double)(truePos+falsePos) : 0.0);
		double recall=(truePos+falseNeg>0? (double)truePos/(double)(truePos+falseNeg) : 0.0);
		double f1=(truePos>0? 2*precision*recall/(precision+recall) : 0.0);
		IOUtil.println("\tPrecision="+truePos+"/"+(truePos+falsePos)+"="+precision);
		IOUtil.println("\tRecall="+truePos+"/"+(truePos+falseNeg)+"="+recall);
		IOUtil.println("\tF1="+f1);
	}
	
	public static String computeAcc(int predLabels[], int trueLabels[])
	{
		int correct=0;
		for (int i=0; i<predLabels.length; i++)
		{
			if (predLabels[i]==trueLabels[i])
			{
				correct++;
			}
		}
		return correct+"/"+predLabels.length+"="+(double)correct/(double)predLabels.length;
	}
	
	public static void printAccuracies(int predLabels[], int labels[], String type, int posLabel, int negLabel)
	{
		IOUtil.println(type+" Accuracy="+computeAcc(predLabels, labels));
		IOUtil.println(type+" Positive PRFs:");
		computePRF(predLabels, labels, posLabel);
		IOUtil.println(type+" Negative PRFs:");
		computePRF(predLabels, labels, negLabel);
	}
	
	public static double computeRMSE(String labelFileName, String outputFileName) throws IOException
	{
		BufferedReader brLabel=new BufferedReader(new InputStreamReader(new FileInputStream(labelFileName), "UTF-8"));
		BufferedReader brPred=new BufferedReader(new InputStreamReader(new FileInputStream(outputFileName), "UTF-8"));
		String labelLine,predLine;
		double error=0.0,label,pred;
		int num=0;
		while ((labelLine=brLabel.readLine())!=null && (predLine=brPred.readLine())!=null)
		{
			num++;
			label=Double.valueOf(labelLine);
			pred=Double.valueOf(predLine);
			error+=(pred-label)*(pred-label);
		}
		brLabel.close();
		brPred.close();
		return Math.sqrt(error/(double)num);
	}
	
	public static double computeRMSE(double labels[], double preds[]) throws IOException
	{
		if (labels.length!=preds.length) return Double.MAX_VALUE;
		double error=0.0;
		for (int i=0; i<labels.length; i++)
		{
			error+=(preds[i]-labels[i])*(preds[i]-labels[i]);
		}
		return Math.sqrt(error/(double)labels.length);
	}
	
	//generation from distribution(s)
	public static double sampleIG(double muIG, double lambdaIG)
	{
		double v=randoms.nextGaussian();   
		double y=v*v;
		double x=muIG+(muIG*muIG*y)/(2*lambdaIG)-(muIG/(2*lambdaIG))*Math.sqrt(4*muIG*lambdaIG*y + muIG*muIG*y*y);
		double test=randoms.nextDouble();
		if (test<=(muIG)/(muIG+x))
		{
			return x;
		}
		return (muIG*muIG)/x;
	}
	
	public static double[] sampleDir(double alpha, int size)
	{
		double alphaVector[]=new double[size];
		for (int i=0; i<size; i++)
		{
			alphaVector[i]=alpha;
		}
		return sampleDir(alphaVector);
	}
	
	public static double[] sampleDir(double alpha[])
	{
		double v[]=new double[alpha.length];
		for (int i=0; i<alpha.length; i++)
		{
			v[i]=randoms.nextGamma(alpha[i]);
			while (v[i]==0.0)
			{
				v[i]=randoms.nextGamma(alpha[i]);
			}
		}
		double sumV=sum(v);
		for (int i=0; i<alpha.length; i++)
		{
			v[i]/=sumV;
		}
		return v;
	}
	
	// compute log (a+b) given log a and log b
	public static double logSum(double logA, double logB)
	{
		double logMax=Math.max(logA, logB);
		return Math.log(Math.exp(logA-logMax)+Math.exp(logB-logMax))+logMax;
	}
	
	public static double[][] invert(double a[][]) 
	{
		int n = a.length;
		double x[][] = new double[n][n];
		double b[][] = new double[n][n];
		int index[] = new int[n];
		for (int i=0; i<n; ++i) 
		{
			b[i][i] = 1;
		} 
	 
		// Transform the matrix into an upper triangle
		gaussian(a, index);
	 
		// Update the matrix b[i][j] with the ratios stored
		for (int i=0; i<n-1; ++i)
		{
			for (int j=i+1; j<n; ++j)
			{
				for (int k=0; k<n; ++k)
				{
					b[index[j]][k]-= a[index[j]][i]*b[index[i]][k];
				}
			}
		}

		// Perform backward substitutions
		for (int i=0; i<n; ++i) 
		{
			x[n-1][i] = b[index[n-1]][i]/a[index[n-1]][n-1];
			for (int j=n-2; j>=0; --j) 
			{
				x[j][i] = b[index[j]][i];
				for (int k=j+1; k<n; ++k) 
				{
					x[j][i] -= a[index[j]][k]*x[k][i];
				}
				x[j][i] /= a[index[j]][j];
			}
		}
		
		return x;
	}
	
	// Method to carry out the partial-pivoting Gaussian
	// elimination.  Here index[] stores pivoting order.
	public static void gaussian(double a[][], int index[]) 
	{
		int n = index.length;
		double c[] = new double[n];
	 
		// Initialize the index
		for (int i=0; i<n; ++i) 
		{
			index[i] = i;
		}
	 
		// Find the rescaling factors, one from each row
		for (int i=0; i<n; ++i) 
		{
			double c1 = 0;
			for (int j=0; j<n; ++j) 
			{
				double c0 = Math.abs(a[i][j]);
				if (c0 > c1) c1 = c0;
			}
			c[i] = c1;
		}
	 
		// Search the pivoting element from each column
		int k = 0;
		for (int j=0; j<n-1; ++j) 
		{
			double pi1 = 0;
			for (int i=j; i<n; ++i) 
			{
				double pi0 = Math.abs(a[index[i]][j]);
				pi0 /= c[index[i]];
				if (pi0 > pi1) 
				{
					pi1 = pi0;
					k = i;
				}
			}
	 
			// Interchange rows according to the pivoting order
			int itmp = index[j];
			index[j] = index[k];
			index[k] = itmp;
			for (int i=j+1; i<n; ++i) 	
			{
				double pj = a[index[i]][j]/a[index[j]][j];
				
				// Record pivoting ratios below the diagonal
				a[index[i]][j] = pj;
	 
				// Modify other elements accordingly
				for (int l=j+1; l<n; ++l)
				{
					 a[index[i]][l] -= pj*a[index[j]][l];
				}  
			}
		}
	}
	
	public static double pearson(double[] x, double[] y)
	{
		if (x.length!=y.length || x.length==0 || y.length==0) return 0.0;
		double mx=average(x),my=average(y),numer=0.0,denom1=0.0,denom2=0.0;
		for (int i=0; i<x.length; i++)
		{
			double temp1=x[i]-mx,temp2=y[i]-my;
			numer+=temp1*temp2;
			denom1+=temp1*temp1;
			denom2+=temp2*temp2;
		}
		return numer/Math.sqrt(denom1*denom2);
	}
}

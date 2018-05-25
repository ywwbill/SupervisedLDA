package util;

import java.io.IOException;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.ArrayList;

public class PCEval
{
	public void pearsonCorrelation(String predFileName, String labelFileName, String resultFileName) throws IOException
	{
		ArrayList<String> strPreds=IOUtil.loadStringList(predFileName);
		ArrayList<String> strLabels=IOUtil.loadStringList(labelFileName);
		if (strPreds.size()!=strLabels.size())
		{
			String msg="Number of lines are different in prediction and gold label files: "+strPreds.size()+" and "+strLabels.size();
			IOUtil.println(msg);
			if (resultFileName!=null && resultFileName.length()>0)
			{
				BufferedWriter bw=new BufferedWriter(new FileWriter(resultFileName));
				bw.write(msg);
				bw.newLine();
				bw.close();
			}
			return;
		}
		
		int len=strLabels.size();
		while (strLabels.get(len-1).equals("0.000")) len--;
		double[] preds=new double[len];
		double[] labels=new double[len];
		for (int i=0; i<len; i++)
		{
			preds[i]=Double.valueOf(strPreds.get(i));
			labels[i]=Double.valueOf(strLabels.get(i));
		}
		double r=MathUtil.pearson(preds, labels);
		IOUtil.println("The pearson correlation is "+r);
		
		int len05=0;
		for (int i=0; i<len; i++)
		{
			if (labels[i]>=0.5) len05++;
		}
		double[] preds05=new double[len05];
		double[] labels05=new double[len05];
		int j=0;
		for (int i=0; i<len; i++)
		{
			if (labels[i]>=0.5)
			{
				preds05[j]=preds[i];
				labels05[j]=labels[i];
				j++;
			}
		}
		double r05=MathUtil.pearson(preds05, labels05);
		IOUtil.println("The pearson correlation for examples>0.5 is "+r05);
		
		if (resultFileName!=null && resultFileName.length()>0)
		{
			BufferedWriter bw=new BufferedWriter(new FileWriter(resultFileName));
			bw.write(r+"");
			bw.newLine();
			bw.write(r05+"");
			bw.newLine();
			bw.close();
		}
	}
}

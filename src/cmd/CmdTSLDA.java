package cmd;

import java.io.IOException;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

import tlda.TLDAParam;
import tlda.med_tslda.MedTSLDA;
import tlda.tslda.TSLDA;
import util.IOUtil;

public class CmdTSLDA
{
	public static void main(String args[]) throws IOException, ParseException
	{
		Options options=new Options();
		options.addOption(new Option("h", "Print this message"));
		//Basic parameters
		options.addOption(new Option("a", true, "(Optinal, default 0.01) Dirichlet parameter for the document distributions over topics"));
		options.addOption(new Option("b", true, "(Optinal, default 0.01) Dirichlet parameter for the topic distributions over words"));
		options.addOption(new Option("k", true, "(Optinal, default 20) Number of topics"));
		options.addOption(new Option("i", true, "(Optinal, default 500) Number of iterations"));
		//Regression parameters
		options.addOption(new Option("mu", true, "(Optinal, default 0.0) The mean of the Gaussian priors for regression parameters"));
		options.addOption(new Option("n", true, "(Optinal, default 1.0) The variance of the Gaussian priors for regression parameters"));
		options.addOption(new Option("c", true, "(Optinal, default 1.0) The regularization parameter for hinge loss"));
		options.addOption(new Option("e", true, "(Optinal, default 0.1) The error bound of hinge loss"));
		options.addOption(new Option("s", true, "(Optinal, default 1.0) The variance of the Gaussian distribution for generating documents' response labels"));
		options.addOption(new Option("hl", "(Optinal, default false) Use hinge loss"));
		//IO parameters
		options.addOption(new Option("t", "(Optinal, default false) Use a pre-trained model to infer on test data"));
		options.addOption(new Option("tp", true, "Tree prior file name"));
		options.addOption(new Option("v", true, "Vocabulary file name"));
		options.addOption(new Option("d", true, "Corpus file name"));
		options.addOption(new Option("l", true, "(Optional when inferring on test data) Label file name"));
		options.addOption(new Option("m", true, "Model file name"));
		//Optional IO parameters
		options.addOption(new Option("r", true, "(Optional) Human-readable result file name"));
		options.addOption(new Option("tc", true, "(Optional) Documents' topic count file name"));
		options.addOption(new Option("w", true, "(Optional, default 20) Number of top words for each topic and for positive/negative weights in the result file"));
		options.addOption(new Option("p", true, "(Optional) Predicted value file name"));
		
		
		CommandLineParser parser=new DefaultParser();
		CommandLine cmd=parser.parse(options, args);
		HelpFormatter format=new HelpFormatter();
		if (cmd.hasOption("h"))
		{
			format.printHelp("TLDA", options);
			return;
		}
		
		if (!cmd.hasOption("v"))
		{
			IOUtil.println("Vocabulary file is not given.");
			format.printHelp("TLDA", options);
			return;
		}
		if (!cmd.hasOption("d"))
		{
			IOUtil.println("Corpus file is not given.");
			format.printHelp("TLDA", options);
			return;
		}
		if (!cmd.hasOption("l") && !cmd.hasOption("t"))
		{
			IOUtil.println("Label file is not given.");
			format.printHelp("TLDA", options);
			return;
		}
		if (!cmd.hasOption("m"))
		{
			IOUtil.println("Model file is not given.");
			format.printHelp("TLDA", options);
			return;
		}
		if (!cmd.hasOption("tp"))
		{
			IOUtil.println("Tree prior file is not given.");
			format.printHelp("TLDA", options);
			return;
		}
		
		TLDAParam param=new TLDAParam(cmd.getOptionValue("v"), cmd.getOptionValue("tp"));
		if (cmd.hasOption("a"))
		{
			double value=Double.valueOf(cmd.getOptionValue("a"));
			if (value>0.0) param.alpha=value;
		}
		if (cmd.hasOption("b"))
		{
			double value=Double.valueOf(cmd.getOptionValue("b"));
			if (value>0.0) param.beta=value;
		}
		if (cmd.hasOption("k"))
		{
			int value=Integer.valueOf(cmd.getOptionValue("k"));
			if (value>0) param.numTopics=value;
		}
		int numIters=500;
		if (cmd.hasOption("i"))
		{
			int value=Integer.valueOf(cmd.getOptionValue("i"));
			if (value>0) numIters=value;
		}
		int numTopWords=20;
		if (cmd.hasOption("w"))
		{
			int value=Integer.valueOf(cmd.getOptionValue("w"));
			if (value>0) numTopWords=value;
		}
		if (cmd.hasOption("mu"))
		{
			param.mu=Double.valueOf(cmd.getOptionValue("mu"));
		}
		if (cmd.hasOption("s"))
		{
			double value=Double.valueOf(cmd.getOptionValue("s"));
			if (value>0.0) param.sigma=value;
		}
		if (cmd.hasOption("n"))
		{
			double value=Double.valueOf(cmd.getOptionValue("n"));
			if (value>0.0) param.nu=value;
		}
		if (cmd.hasOption("c"))
		{
			double value=Double.valueOf(cmd.getOptionValue("c"));
			if (value>0.0) param.c=value;
		}
		if (cmd.hasOption("e"))
		{
			double value=Double.valueOf(cmd.getOptionValue("e"));
			if (value>0.0) param.eps=value;
		}
		
		TSLDA slda;
		if (!cmd.hasOption("t"))//train
		{
			if (!cmd.hasOption("hl"))//mse
			{
				slda=new TSLDA(param);
			}
			else//hinge
			{
				slda=new MedTSLDA(param);
			}
			slda.readCorpus(cmd.getOptionValue("d"));
			slda.readLabels(cmd.getOptionValue("l"));
			slda.initialize();
			slda.sample(numIters);
			slda.writeModel(cmd.getOptionValue("m"));
			if (cmd.hasOption("r")) slda.writeResult(cmd.getOptionValue("r"), numTopWords);
			if (cmd.hasOption("tc")) slda.writeDocTopicCounts(cmd.getOptionValue("tc"));
			if (cmd.hasOption("p")) slda.writePredLabels(cmd.getOptionValue("p"));
		}
		else//test
		{
			if (!cmd.hasOption("hl"))//mse
			{
				slda=new TSLDA(cmd.getOptionValue("m"), param);
			}
			else//hinge
			{
				slda=new MedTSLDA(cmd.getOptionValue("m"), param);
			}
			slda.readCorpus(cmd.getOptionValue("d"));
			if (cmd.hasOption("l")) slda.readLabels(cmd.getOptionValue("l"));
			slda.initialize();
			slda.sample(numIters);
			if (cmd.hasOption("tc")) slda.writeDocTopicCounts(cmd.getOptionValue("tc"));
			if (cmd.hasOption("p")) slda.writePredLabels(cmd.getOptionValue("p"));
		}
	}
}

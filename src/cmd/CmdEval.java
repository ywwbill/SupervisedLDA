package cmd;

import java.io.IOException;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

import util.IOUtil;
import util.PCEval;

public class CmdEval
{
	public static void main(String args[]) throws IOException, ParseException
	{
		Options options=new Options();
		options.addOption(new Option("h", "Print this message"));
		options.addOption(new Option("p", true, "Prediction file name"));
		options.addOption(new Option("l", true, "Gold label file name"));
		options.addOption(new Option("o", true, "(Optional, default null) Write the pearson correlation to the given file"));
		
		CommandLineParser parser=new DefaultParser();
		CommandLine cmd=parser.parse(options, args);
		HelpFormatter format=new HelpFormatter();
		if (cmd.hasOption("h"))
		{
			format.printHelp("Pearson Correlation", options);
			return;
		}
		
		if (!cmd.hasOption("p"))
		{
			IOUtil.println("Prediction file is not given.");
			format.printHelp("Pearson Correlation", options);
			return;
		}
		if (!cmd.hasOption("l"))
		{
			IOUtil.println("Gold label file is not given.");
			format.printHelp("Pearson Correlation", options);
			return;
		}
		
		String outputFN=null;
		if (cmd.hasOption("o"))
		{
			outputFN=cmd.getOptionValue("o");
		}
		PCEval eval=new PCEval();
		eval.pearsonCorrelation(cmd.getOptionValue("p"), cmd.getOptionValue("l"), outputFN);
	}
}

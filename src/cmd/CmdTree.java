package cmd;

import java.io.IOException;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;

import tlda.TreeBuilder;
import util.IOUtil;

public class CmdTree
{
	public static void main(String args[]) throws IOException, ParseException
	{
		Options options=new Options();
		options.addOption(new Option("h", "Print this message"));
		options.addOption(new Option("v", true, "Vocabulary file name"));
		options.addOption(new Option("e", true, "Word embedding file name"));
		options.addOption(new Option("o", true, "Tree prior file name"));
		options.addOption(new Option("t", true, "(Optional, default 1) Tree prior type: 1-Two level tree 2-HAC 3-HAC with leaf duplication"));
		options.addOption(new Option("k", true, "(Optional, default 10) Number of child nodes per internal node for a two-level tree"));
		
		CommandLineParser parser=new DefaultParser();
		CommandLine cmd=parser.parse(options, args);
		HelpFormatter format=new HelpFormatter();
		if (cmd.hasOption("h"))
		{
			format.printHelp("Tree Prior", options);
			return;
		}
		
		if (!cmd.hasOption("v"))
		{
			IOUtil.println("Vocabulary file is not given.");
			format.printHelp("Tree Prior", options);
			return;
		}
		if (!cmd.hasOption("e"))
		{
			IOUtil.println("Word embedding file is not given.");
			format.printHelp("Tree Prior", options);
			return;
		}
		if (!cmd.hasOption("o"))
		{
			IOUtil.println("Tree prior file is not given.");
			format.printHelp("Tree Prior", options);
			return;
		}
		
		int treeType=1;
		if (cmd.hasOption("t"))
		{
			int value=Integer.valueOf(cmd.getOptionValue("t"));
			if (value==1 || value==2 || value==3) treeType=value;
		}
		int numTop=10;
		if (cmd.hasOption("k"))
		{
			int value=Integer.valueOf(cmd.getOptionValue("k"));
			if (value>0) numTop=value;
		}
		
		TreeBuilder tb=new TreeBuilder();
		tb.buildTree(cmd.getOptionValue("v"), cmd.getOptionValue("e"), cmd.getOptionValue("o"), treeType, numTop);
	}
}

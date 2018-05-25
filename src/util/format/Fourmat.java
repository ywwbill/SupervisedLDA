package util.format;

import java.text.NumberFormat;
import java.util.Arrays;

public class Fourmat
{
	private static NumberFormat format;
	
	static
	{
		format=NumberFormat.getInstance();
		format.setMaximumFractionDigits(4);
		format.setMinimumFractionDigits(4);
		format.setGroupingUsed(false);
	}
	
	public static String format(double nums[])
	{
		String formatStr[]=new String[nums.length];
		for (int i=0; i<nums.length; i++)
		{
			formatStr[i]=format(nums[i]);
		}
		return Arrays.toString(formatStr);
	}
	
	public static String format(double num)
	{
		if (Double.isNaN(num)) return "NaN";
		return format.format(num);
	}
}

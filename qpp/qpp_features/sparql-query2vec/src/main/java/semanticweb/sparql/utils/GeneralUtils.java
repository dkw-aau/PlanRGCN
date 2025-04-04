package semanticweb.sparql.utils;

import semanticweb.sparql.config.ProjectConfiguration;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.io.RandomAccessFile;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Scanner;

public class GeneralUtils {
	public static void saveExecutionTimePredictions(double[] y, String fileName) throws IOException {
		PrintStream ps = new PrintStream(fileName);
		ps.println(ProjectConfiguration.getExecutionTimeHeader());
		
		
		
		for(double yi:y) {
			ps.println(yi);
		}
		ps.close();
		
	}
	
	public static List<String> loadQuries(String queryFile) throws IOException {
		List<String> queries;
		FileInputStream fis = new FileInputStream(queryFile);
		Scanner in = new Scanner(fis);
		queries = null;
		queries = new ArrayList<String>();
		boolean ignoredHeader = false;
		int count = 0;
		List<Integer> removequeries = Arrays.asList( 1304, 2057, 5409, 7203, 7204, 2557, 2558, 4900, 8630, 9030, 9031, 9547, 2497, 4994, 7491, 9987);
		while(in.hasNext()) {

			String line = in.nextLine();
			//System.out.println(line);
			if(!ignoredHeader){
				ignoredHeader = true;
			}
			else if(!removequeries.contains(count))
				queries.add(line.split(",")[1].replaceAll("\"",""));
				System.out.println(count);
			count++;
		}	
		fis.close();
		
		return queries;
	}
	
	public static void addHeader(String file, String header) throws IOException {
		RandomAccessFile f = new RandomAccessFile(file, "rw");
		f.seek(0); // to the beginning
		f.write((header+"\n").getBytes());
		f.close();
	}
}

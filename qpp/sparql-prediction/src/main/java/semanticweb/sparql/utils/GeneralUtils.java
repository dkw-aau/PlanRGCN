package semanticweb.sparql.utils;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintStream;
import java.io.RandomAccessFile;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

import semanticweb.sparql.config.ProjectConfiguration;

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
		
		while(in.hasNext()) {
			String line = in.nextLine();
			//System.out.println(line);
			queries.add(line);
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

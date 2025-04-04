package semanticweb.sparql.preprocess;

import org.apache.commons.lang3.time.StopWatch;
import ged.AlgorithmConfig;
import semanticweb.RDFGraphMatching;
import semanticweb.sparql.config.ProjectConfiguration;
import semanticweb.sparql.utils.DBPediaUtils;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintStream;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;
import java.util.Scanner;
import java.util.concurrent.TimeUnit;

public class SparqlDistance {
	//public static String CONFIG_FILE = System.getProperty("user.home")+"/Documents/code/query-performance/config-6000.prop";
	//public static String CONFIG_FILE = System.getProperty("user.home")+"/Documents/code/query-performance/config-knn-dbpsb-test.prop";
	//public static String CONFIG_FILE = System.getProperty("user.home")+"/Documents/code/query-performance/config-dbpsb-k10.prop";
	//public static String CONFIG_FILE = System.getProperty("user.home")+"/Documents/code/query-performance/config-dbpsb-k5.prop";
	//public static String CONFIG_FILE = System.getProperty("user.home")+"/Documents/code/query-performance/config-dbpsb-k15.prop";
	//public static String CONFIG_FILE = System.getProperty("user.home")+"/Documents/code/query-performance/config-dbpsb-k20.prop";
	//public static String CONFIG_FILE = System.getProperty("user.home")+"/Documents/code/query-performance/config-dbpsb-k25.prop";
	
	String trainingQueryMatrixFilename = "training_query_matrix.nogit";
	String trainingQueryHungarianFilename = null;
	
	private Properties prop;
	List<String> queries;
	public SparqlDistance() throws FileNotFoundException, IOException {
		
		prop = new Properties();
		prop.load(new FileInputStream(ProjectConfiguration.CONFIG_FILE));
		//System.out.println(prop.getProperty("TestQuerySmall"));
		//System.out.println(prop.values());
		
		loadConfig();
		loadQuries();
		
	}
	
	public void loadConfig() {
		trainingQueryHungarianFilename = prop.getProperty("TrainingDistanceHungarianMatrix");
	}
	
	public void loadQuries() throws IOException {
		String queryFile = getProperty("TrainingQuery");
		System.out.println("TrainingQuery:"+queryFile);
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
		
	}
	
	public void generateQuerySparseMatrix(String queryFile,String matrixFile) throws IOException{
		
		PrintStream ps = new PrintStream(matrixFile);
		
		for(int i=0;i<queries.size();i++) {
			String iQuery = queries.get(i);
			for(int j=i+1;j<queries.size();j++) {
				String jQuery = queries.get(j);
				String out = i+" "+j+" "+iQuery+" "+jQuery;
				//System.out.println(out);
				ps.println(out);
				
			}
		}
		
		ps.close();
	}
	
	public void generateQueryHungarianDistanceSparseMatrix(String queryMatrixFile,String distanceMatrixFile) throws Exception{
		FileInputStream fis = new FileInputStream(queryMatrixFile);
		PrintStream ps = new PrintStream(distanceMatrixFile);
		Scanner in = new Scanner(fis);
		DecimalFormat df=new DecimalFormat("0.000");
		
		PrintStream ps_error = new PrintStream("error.txt");
		
		//List<String> queryPairs = new ArrayList<String>();
		int count = 0;
		StopWatch watch = new StopWatch();
		watch.start();
		while(in.hasNext()) {
			if(count%10000 == 0) {
				System.out.println((count)+" query pairs processed");
				System.out.println("Time taken: "+watch.getTime()+" seconds");
			}
			String line = in.nextLine();
			String[] vals = line.split(" ");
			//System.out.println("i:"+vals[0]);
			//System.out.println("j:"+vals[1]);
			String q1 = vals[2];
			String q2 = vals[3];
			//System.out.println("q1: "+q1);
			//System.out.println("q2: "+q2);
			String sparql1 = DBPediaUtils.getParam(q1, "query");
			String sparql2 = DBPediaUtils.getParam(q2, "query");
			
			double cost = Integer.MAX_VALUE;
			
			if(sparql1.isEmpty()==false && sparql2.isEmpty()==false) {
				String rsparql1 = DBPediaUtils.refineForDBPedia(sparql1);
				String rsparql2 = DBPediaUtils.refineForDBPedia(sparql2);
				AlgorithmConfig algorithmConfig = AlgorithmConfig.createBipartiteHungarian();
				RDFGraphMatching matcher = new RDFGraphMatching();
				//System.out.println("q1: "+sparql1);
				//System.out.println("q2: "+sparql2);
				
				cost = matcher.queryGraphDistance(rsparql1, rsparql2, algorithmConfig);
				//System.out.println("Cost:"+cost);
			}
			String out = vals[0]+" "+vals[1]+" "+df.format(cost);
			ps.println(out);
			count++;
			
			if(cost==Integer.MAX_VALUE) {
				String ss = "#######################\n"+vals[0]+" "+vals[1]+"\n--------q1-below--------\n"+q1+"\n"+sparql1+"\n-------q2-below---------\n"+q2+"\n"+sparql2+"\n------end-----\n";
				ps_error.println(ss);
			}
		}
		ps.close();
		ps_error.close();


	}

	
	public void generateQueryHungarianDistanceSparseMatrix(String distanceMatrixFile) throws Exception{
		
		PrintStream ps = new PrintStream(distanceMatrixFile);
		
		DecimalFormat df=new DecimalFormat("0.000");
		
		PrintStream ps_error = new PrintStream("error.txt");
		System.out.println("Total training queries:"+queries.size());
		//List<String> queryPairs = new ArrayList<String>();
		int count = 0;
		StopWatch watch = new StopWatch();
		watch.start();
		
		for(int i=0;i<queries.size();i++) {
			String q1 = queries.get(i);
			for(int j=i+1;j<queries.size();j++) {
				String q2 = queries.get(j);

				
				if(count%10000 == 0) {
					System.out.println((count)+" query pairs processed");
					System.out.println("Time taken: "+watch.getTime()+" seconds");
				}
				
				String sparql1 = DBPediaUtils.getParam(q1, "query");
				String sparql2 = DBPediaUtils.getParam(q2, "query");
				
				double cost = Integer.MAX_VALUE;
				//System.out.println("q1: "+sparql1);
				//System.out.println("q2: "+sparql2);

				if (sparql1.isEmpty() == false && sparql2.isEmpty() == false) {
					String rsparql1 = DBPediaUtils.refineForDBPedia(sparql1);
					String rsparql2 = DBPediaUtils.refineForDBPedia(sparql2);
					AlgorithmConfig algorithmConfig = AlgorithmConfig.createBipartiteHungarian();
					RDFGraphMatching matcher = new RDFGraphMatching();

					cost = matcher.queryGraphDistance(rsparql1, rsparql2, algorithmConfig);
					//System.out.println("Cost:"+cost);
				}
				String out = i+" "+j+" "+df.format(cost);
				ps.println(out);
				count++;
				
				if(cost==Integer.MAX_VALUE) {
					String ss = "#######################\n"+i+" "+j+"\n--------q1-below--------\n"+q1+"\n"+sparql1+"\n-------q2-below---------\n"+q2+"\n"+sparql2+"\n------end-----\n";
					ps_error.println(ss);
				}
			}
		}	
		System.out.println("Total "+(count)+" query pairs processed");
		ps.close();
		ps_error.close();


	}	
	public static double dbpediaQueryGraphHumgarainDistance(String q1, String q2) throws Exception {
		double cost = Integer.MAX_VALUE;
		String sparql1 = DBPediaUtils.getParam(q1, "query");
		String sparql2 = DBPediaUtils.getParam(q2, "query");		
		if(sparql1.isEmpty()==false && sparql2.isEmpty()==false) {
			String rsparql1 = DBPediaUtils.refineForDBPedia(sparql1);
			String rsparql2 = DBPediaUtils.refineForDBPedia(sparql2);
			AlgorithmConfig algorithmConfig = AlgorithmConfig.createBipartiteHungarian();
			RDFGraphMatching matcher = new RDFGraphMatching();
			//System.out.println("q1: "+sparql1);
			//System.out.println("q2: "+sparql2);
			
			cost = matcher.queryGraphDistance(rsparql1, rsparql2, algorithmConfig);
			//System.out.println("Cost:"+cost);
		}
		return cost;
	}
	
	public String getProperty(String x) {
		return prop.getProperty(x);
	}
	
	public String getProperty(String x,String defaultVal) {
		return prop.getProperty(x,defaultVal);
	}
	
	public void processTrainingQueries() throws Exception {

		StopWatch watch = new StopWatch();
		watch.start();
		//System.out.println("Generating query sparse matrix file");
		//generateQuerySparseMatrix(getProperty("TrainingQuery"),trainingQueryMatrixFilename);
		//System.out.println("Elapsed time: "+watch.elapsed(TimeUnit.SECONDS)+" seconds");
		
		watch.reset();
		watch.start();
		System.out.println("Generating hungarian distance matrix file");
		generateQueryHungarianDistanceSparseMatrix(trainingQueryHungarianFilename);
		watch.stop();
		System.out.println("Elapsed time: "+watch.getTime()+" seconds");
	}

	public static void main(String[] args) throws Exception {
		SparqlDistance sd = new SparqlDistance();
		sd.processTrainingQueries();
		/*sd.generateQuerySparseMatrix(sd.getProperty("TestQuerySmall"),"query_matrix.dat");
		StopWatch watch = new StopWatch();

		watch.start();
		sd.generateQueryHungarianDistanceSparseMatrix("query_matrix.dat", "distance_matrix.dat");
		watch.stop();
		System.out.println("Elapsed time:"+watch.elapsed(TimeUnit.SECONDS));*/
	}
}

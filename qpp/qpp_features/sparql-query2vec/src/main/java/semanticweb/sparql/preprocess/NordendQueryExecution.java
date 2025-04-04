package semanticweb.sparql.preprocess;

import org.apache.commons.lang3.time.StopWatch;
import org.apache.jena.query.*;
import semanticweb.sparql.utils.DBPediaUtils;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Scanner;

public class NordendQueryExecution {
	
	private static String ENDPOINT = "https://dbpedia.org/sparql";
	private static String QUERIES = "dbp-100-random.log";
	
	public static ResultSet queryTDB(String qStr) {
		String q = DBPediaUtils.refineForDBPedia(qStr);
		Query query = QueryFactory.create(q);
		QueryExecution qexec = QueryExecutionFactory.sparqlService(ENDPOINT, query);
		ResultSet results = qexec.execSelect();
		return results;

	}
	
	public static void executeQueries() throws IOException {
		PrintStream psTime = new PrintStream("nordend_ex_time");
		PrintStream psRec = new PrintStream("nordend_record_count");
		PrintStream psQuery = new PrintStream("nordend_query");
		
		FileInputStream fis = new FileInputStream(QUERIES);
		Scanner in = new Scanner(fis);
		
		StopWatch watch = new StopWatch();
		watch.start();		
		int count = 0;
		int goodQueriesCount = 0;
		
		while(in.hasNext()) {
			//System.out.println("Processing query:"+count);
			

			
			String line = in.nextLine();
			String[] ss = line.split(" ");
			String q = ss[6].substring(1, ss[6].length()-1);
			
			//System.out.println(line);
			//queries.add(line);
	

			String qStr = DBPediaUtils.getQueryForDBpedia(q);
			

			
			watch.reset();
			watch.start();
			try {
				
				ResultSet results = queryTDB(qStr);
				long elapsed = watch.getTime();


				ResultSetRewindable rsrw = ResultSetFactory.copyResults(results);
			    int numberOfResults = rsrw.size();
			    if(numberOfResults>0) {
					psTime.println(elapsed);
					psQuery.println(q);
				    psRec.println(numberOfResults);
				    goodQueriesCount++;
				
					
			    }
				
		    
			} catch(Exception e) {
				//do nothing
			}
			
			count++;
			if(count%1000==0) {
				System.out.println(count+" queries processed");
			}
		}
		
		
		System.out.println(goodQueriesCount+" queries selected");
		
		psTime.close();
		psRec.close();
		psQuery.close();
		
		
		fis.close();		
		

	}
	
	public static void main(String[] args) throws IOException {
		
		
		if(args.length>0) {
		
			if(args.length!=2) {
				System.out.println("Plsease provide the endpoint and query file");
				return;
			}
			ENDPOINT = args[0];
			QUERIES = args[1];
			
		} 
			
		System.out.println("Endpoint: "+ENDPOINT+" and queries file: "+QUERIES);
	
		
		//"/home/daniel/Documentos/ML/rhassan/query-performance/dbp-100-random.log"
		executeQueries();
	}
}

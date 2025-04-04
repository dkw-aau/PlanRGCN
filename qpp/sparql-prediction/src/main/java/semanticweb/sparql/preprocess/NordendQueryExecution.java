package semanticweb.sparql.preprocess;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Scanner;
import java.util.concurrent.TimeUnit;

/*import com.google.common.base.Stopwatch;
import com.hp.hpl.jena.query.Query;
import com.hp.hpl.jena.query.QueryExecution;
import com.hp.hpl.jena.query.QueryExecutionFactory;
import com.hp.hpl.jena.query.QueryFactory;
import com.hp.hpl.jena.query.ResultSet;
import com.hp.hpl.jena.query.ResultSetFactory;
import com.hp.hpl.jena.query.ResultSetRewindable;
import com.google.common.base.Stopwatch;*/
import org.apache.jena.ext.com.google.common.base.Stopwatch;
import org.apache.jena.query.Query;
import org.apache.jena.query.QueryExecution;
import org.apache.jena.query.QueryExecutionFactory;
import org.apache.jena.query.QueryFactory;
import org.apache.jena.query.ResultSet;
import org.apache.jena.query.ResultSetFactory;
import org.apache.jena.query.ResultSetRewindable;

import semanticweb.sparql.config.ProjectConfiguration;
import semanticweb.sparql.utils.DBPediaUtils;

public class NordendQueryExecution {
	
	private static String ENDPOINT = "http://localhost:3030/dbpedia/sparql";
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
		
		Stopwatch watch = Stopwatch.createStarted();
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
				long elapsed = watch.elapsed(TimeUnit.MILLISECONDS);


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
	
		
		//"/Users/hrakebul/Documents/code/query-performance/dbp-100-random.log"
		executeQueries();
	}
}

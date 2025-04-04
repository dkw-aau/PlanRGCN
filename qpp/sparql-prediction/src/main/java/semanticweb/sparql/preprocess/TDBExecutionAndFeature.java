package semanticweb.sparql.preprocess;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;
import java.util.Scanner;
import java.util.concurrent.TimeUnit;

//import org.apache.jena.ext.com.google.common.base.Stopwatch;
//import org.apache.jena.ext.com.google.common.base.Stopwatch;
import semanticweb.sparql.config.ProjectConfiguration;
import semanticweb.sparql.utils.DBPediaUtils;
import semanticweb.sparql.utils.GeneralUtils;

import com.google.common.base.Stopwatch;
import org.apache.jena.query.*;
/*import com.hp.hpl.jena.query.Dataset;
import com.hp.hpl.jena.query.Query;
import com.hp.hpl.jena.query.QueryExecution;
import com.hp.hpl.jena.query.QueryExecutionFactory;
import com.hp.hpl.jena.query.QueryFactory;
import com.hp.hpl.jena.query.ResultSet;
import com.hp.hpl.jena.query.ResultSetFactory;
import com.hp.hpl.jena.query.ResultSetRewindable;*/
//import com.hp.hpl.jena.rdf.model.Model;
import org.apache.jena.rdf.model.Model;
//import com.hp.hpl.jena.tdb.TDBFactory;
import org.apache.jena.tdb.TDBFactory;

public class TDBExecutionAndFeature {
	
	private List<String> trainingQueries;
	private List<String> validationQueries;
	private List<String> testQueries;
	private Properties prop;
	Model model;
	boolean directTDB = false;
	
	public TDBExecutionAndFeature() throws IOException {
		prop = new Properties();
		prop.load(new FileInputStream(ProjectConfiguration.CONFIG_FILE));
		
		//loadAllQueries();
		
	}
	public void loadAllQueries() throws IOException {
		loadTrainingQueries();		
		loadValidationQueries();
		loadTestQueries();		
	}
	
	public void loadTrainingQueries() throws IOException {
		String trainingQueryFile = prop.getProperty("TrainingQuery");
		trainingQueries = GeneralUtils.loadQuries(trainingQueryFile);
	}

	public void loadValidationQueries() throws IOException {
		String validationQueryFile = prop.getProperty("ValidationQuery");
		validationQueries = GeneralUtils.loadQuries(validationQueryFile);
	}
	public void loadTestQueries() throws IOException {
		String testQueryFile = prop.getProperty("TestQuery");
		testQueries = GeneralUtils.loadQuries(testQueryFile);
	}	
	

	
	public ResultSet queryTDB(String qStr) {
		String q = DBPediaUtils.refineForDBPedia(qStr);
		Query query = QueryFactory.create(q);
		QueryExecution qexec = directTDB==true? QueryExecutionFactory.create(query, model): QueryExecutionFactory.sparqlService(prop.getProperty("Endpoint"), query);
		ResultSet results = qexec.execSelect();
		return results;

	}
	
	public void initTDB() {
		String assemblerFile = prop.getProperty("TDBAssembly");
		System.out.println(assemblerFile);
		Dataset dataset = TDBFactory.assembleDataset(assemblerFile) ;
		//model = TDBFactory.assembleModel(assemblerFile) ;
		model = dataset.getDefaultModel();
		
		
	}
	
	public void closeModel() {
		model.close();
	}
	
	private void executeQueries(List<String> queries, String timeOutFile, String recCountOutFile) throws IOException {
		PrintStream psTime = new PrintStream(timeOutFile);
		PrintStream psRec = new PrintStream(recCountOutFile);
		
		Stopwatch watch = Stopwatch.createStarted();
		watch.start();
		
		
		int count = 0;
		for(String q:queries) {
			String qStr = DBPediaUtils.getQueryForDBpedia(q);
			if(count%1000==0) {
				System.out.println(count+" queries processed");
			}
			
			watch.reset();
			watch.start();
			ResultSet results = queryTDB(qStr);
			psTime.println(watch.elapsed(TimeUnit.MILLISECONDS));

			//ResultSetRewindable rsrw = ResultSetFactory.copyResults(results);
			//int numberOfResults = rsrw.size();
			//psRec.println(numberOfResults);
			
			count++;
		}
		psTime.close();
		psRec.close();
	}
	
	public void executeTrainingQueries() throws IOException {
		System.out.println("Processing training queries");
		
		executeQueries(trainingQueries,prop.getProperty("TDBTrainingExecutionTime"),prop.getProperty("TDBTrainingRecordCount"));
		
	}

	public void executeValidationQueries() throws IOException {
		System.out.println("Processing validation queries");
		
		executeQueries(validationQueries,prop.getProperty("TDBValidationExecutionTime"),prop.getProperty("TDBValidationRecordCount"));
		
	}
	
	public void executeTestQueries() throws IOException {
		System.out.println("Processing test queries");
		
		executeQueries(testQueries,prop.getProperty("TDBTestExecutionTime"),prop.getProperty("TDBTestRecordCount"));
		
	}	
	
	public void executeDirectTDB() {
		
		directTDB = true;
		initTDB();
		queryTDB("select * where {<http://dbpedia.org/resource/Berlin> ?p ?o}");
		
		try {
			executeTrainingQueries();
			executeValidationQueries();
			executeTestQueries();
		}
		catch(Exception e) {
			e.printStackTrace();
		}
		
		
		closeModel();
		
	}
	
	public void fusekiTDB() {
		directTDB = false;
		queryTDB("select * where {<http://dbpedia.org/resource/Berlin> ?p ?o}");
		try {
			executeTrainingQueries();
			executeValidationQueries();
			executeTestQueries();
		}
		catch(Exception e) {
			e.printStackTrace();
		}		
	}
	
	
	
	public void executeRandomlySelectedQueries() throws IOException {
		PrintStream psTime = new PrintStream(prop.getProperty("TDBTrainingExecutionTime"));
		PrintStream psRec = new PrintStream(prop.getProperty("TDBTrainingRecordCount"));
		PrintStream psQuery = new PrintStream(prop.getProperty("TrainingQuery"));
		
		
		
		
		Stopwatch watch = Stopwatch.createStarted();
		watch.start();
		
		
		int count = 0;
		FileInputStream fis = new FileInputStream(prop.getProperty("QueryFile"));
		//FileInputStream fis = new FileInputStream(prop.getProperty("TestQuerySmall"));
		
		Scanner in = new Scanner(fis);
		
		int totalQuery = Integer.parseInt(prop.getProperty("TotalQuery"));
		
		int dataSplit = (int) (totalQuery * 0.6);
		int validationSplit = (int) (totalQuery * 0.2);
		int testSplit = (int) (totalQuery * 0.2);
		
		while(in.hasNext()) {
			//System.out.println("Processing query:"+count);
			
			if(count>=totalQuery) break;
			
			if(count == dataSplit) {
				System.out.println("initilizing validation files");
				psTime.close();
				psRec.close();
				psQuery.close();
				
				psQuery = new PrintStream(prop.getProperty("ValidationQuery"));
				
				psTime = new PrintStream(prop.getProperty("TDBValidationExecutionTime"));
				psRec = new PrintStream(prop.getProperty("TDBValidationRecordCount"));				
				
			} else if(count== (dataSplit+validationSplit)) {
				System.out.println("initilizing test files");
				psTime.close();
				psRec.close();
				psQuery.close();
				
				psQuery = new PrintStream(prop.getProperty("TestQuery"));
				
				psTime = new PrintStream(prop.getProperty("TDBTestExecutionTime"));
				psRec = new PrintStream(prop.getProperty("TDBTestRecordCount"));					
			}
			
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
				
					count++;
			    }
				
				if(count%1000==0) {
					System.out.println(count+" queries processed");
				}			    
			} catch(Exception e) {
				//do nothing
			}

		}
		
		
		psTime.close();
		psRec.close();
		psQuery.close();
		
		
		fis.close();

	}
	
	public void fusekiTDBRandomlySelectedQueries() throws IOException {
		directTDB = false;
		executeRandomlySelectedQueries();
	}
	
	public void generateAlgebraFeatureDataset() throws IOException {
		loadAllQueries();
		PrintStream psTraining = new PrintStream(prop.getProperty("TrainingAlgebraFeatures"));
		PrintStream psValidation = new PrintStream(prop.getProperty("ValidationAlgebraFeatures"));
		PrintStream psTest = new PrintStream(prop.getProperty("TestAlgebraFeatures"));
		
		psTraining.println(ProjectConfiguration.getAlgebraFeatureHeader());
		psValidation.println(ProjectConfiguration.getAlgebraFeatureHeader());
		psTest.println(ProjectConfiguration.getAlgebraFeatureHeader());
		
		generateAlgebraFeatures(psTraining, trainingQueries);
		generateAlgebraFeatures(psValidation, validationQueries);
		generateAlgebraFeatures(psTest, testQueries);
		
	}
	
	private void generateAlgebraFeatures(PrintStream ps, List<String> queries) throws IOException {
		
		
		AlgebraFeatureExtractor fe = new AlgebraFeatureExtractor();
		
		for(String q:queries) {
			String queryStr = DBPediaUtils.getQueryForDBpedia(q);
			//System.out.println(q);
			//System.out.println(queryStr);
			double[] features = fe.extractFeatures(queryStr);
			
			for(int i=0;i<features.length;i++) {
				if(i!=0) {
					ps.print(",");
				}
				ps.print(features[i]);
			}
			ps.println();
		}
		
		ps.close();
	}
	
	public static void main(String[] args) throws Exception {
		
		Stopwatch watch = Stopwatch.createStarted();
		watch.start();		
		TDBExecutionAndFeature wrapper = new TDBExecutionAndFeature();
		//wrapper.fusekiTDBRandomlySelectedQueries();
		
		//only run once
		//SparqlDistance sd = new SparqlDistance();
		//sd.processTrainingQueries();
		
		wrapper.generateAlgebraFeatureDataset();
		watch.stop();
		System.out.println("Total time for algebra query extraction: "+watch.elapsed(TimeUnit.MILLISECONDS)+" ms");
		
	}
	
}

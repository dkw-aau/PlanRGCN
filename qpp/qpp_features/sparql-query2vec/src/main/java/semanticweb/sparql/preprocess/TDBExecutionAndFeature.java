package semanticweb.sparql.preprocess;

import org.apache.commons.lang3.time.StopWatch;
import org.apache.jena.query.*;
import org.apache.jena.rdf.model.Model;
import org.apache.jena.tdb.TDBFactory;

import lsq_extract.util.QueryReaderLegacy;
import lsq_extract.util.QueryReaderWAlgRunTime;
import semanticweb.sparql.SparqlUtils;
import semanticweb.sparql.config.ProjectConfiguration;
import semanticweb.sparql.utils.DBPediaUtils;

import java.io.*;
import java.util.*;

public class TDBExecutionAndFeature {
	
	private List<String[]> trainingQueries;
	private List<String> validationQueries;
	private List<String> testQueries;
	private Properties prop;
	private String inputQueryFile, outputFile, input_delimiter, output_delimiter;
	private int idColumn,  queryColumn, execTimeColumn;
	private Model model;
	private boolean directTDB = false;

	public TDBExecutionAndFeature() throws IOException {
		prop = new Properties();
		prop.load(new FileInputStream(ProjectConfiguration.CONFIG_FILE));
		
		//loadAllQueries();
		
	}
	public TDBExecutionAndFeature(String config_file) throws IOException {
		prop = new Properties();
		prop.load(new FileInputStream(config_file));
	}

	/**
	 * Create the proccessing class for algebra features.
	 * @param inputQueryFile Url for imput file that contain queries ins csv format
	 * @param outputFile Url for output file that will contain queries features in csv format
	 * @param configFile Config file, some configs could be defines like Namespaces, Endpoint
	 * @param input_delimiter input delimiter character as string
	 * @param output_delimiter output delimiter character as string
	 * @param idColumn index of id of query in array that will be parsed by row.
	 * @param queryColumn  index of query string in array that will be parsed by row.
	 * @param execTimeColumn  index of  execution time in array that will be parsed by row.
	 * @throws IOException Error on file read
	 */
	public TDBExecutionAndFeature(String inputQueryFile, String outputFile, String prefixFile, String input_delimiter, String output_delimiter, int idColumn, int queryColumn, int execTimeColumn) throws IOException {
		prop = new Properties();
		//prop.load(new FileInputStream(configFile));
		if(!prefixFile.isEmpty()){
			this.prop.setProperty("Namespaces", prefixFile);
		}
		this.inputQueryFile= inputQueryFile;
		this.outputFile= outputFile;
		this.input_delimiter= input_delimiter;
		this.output_delimiter= output_delimiter;
		this.idColumn= idColumn;
		this.queryColumn= queryColumn;
		this.execTimeColumn= execTimeColumn;
	}

	public ResultSet queryTDB(String qStr) {
		String q = DBPediaUtils.refineForDBPedia(qStr);
		Query query = QueryFactory.create(q);
		QueryExecution qexec = directTDB ? QueryExecutionFactory.create(query, model): QueryExecutionFactory.sparqlService(prop.getProperty("Endpoint"), query);
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
	
	private void executeQueries(List<String[]> queries, String timeOutFile, String recCountOutFile) throws IOException {
		PrintStream psTime = new PrintStream(timeOutFile);
		PrintStream psRec = new PrintStream(recCountOutFile);
		
		StopWatch watch = new StopWatch();
		watch.start();
		
		
		int count = 0;
		for(String[] q:queries) {
			String qStr = DBPediaUtils.getQueryForDBpedia(q[1]);
			if(count%1000==0) {
				System.out.println(count+" queries processed");
			}
			
			watch.reset();
			watch.start();
			ResultSet results = queryTDB(qStr);
			psTime.println(watch.getTime());
			count++;
		}
		psTime.close();
		psRec.close();
	}
	
	public void executeTrainingQueries() throws IOException {
		System.out.println("Processing training queries");
		
		executeQueries(trainingQueries, prop.getProperty("TDBTrainingExecutionTime"),prop.getProperty("TDBTrainingRecordCount"));
		
	}
	
	public void executeDirectTDB() {
		
		directTDB = true;
		initTDB();
		queryTDB("select * where {<http://dbpedia.org/resource/Berlin> ?p ?o}");
		
		try {
			executeTrainingQueries();
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
		}
		catch(Exception e) {
			e.printStackTrace();
		}		
	}

	public void executeRandomlySelectedQueries() throws IOException {
		PrintStream psTime = new PrintStream(prop.getProperty("TDBTrainingExecutionTime"));
		PrintStream psRec = new PrintStream(prop.getProperty("TDBTrainingRecordCount"));
		PrintStream psQuery = new PrintStream(prop.getProperty("TrainingQuery"));
		
		
		
		
		StopWatch watch = new StopWatch();
		watch.start();
		
		
		int count = 0;
		FileInputStream fis = new FileInputStream(prop.getProperty("QueryFile"));

		Scanner in = new Scanner(fis);
		
		int totalQuery = Integer.parseInt(prop.getProperty("TotalQuery"));
		
		int dataSplit = (int) (totalQuery * 0.6);
		int validationSplit = (int) (totalQuery * 0.2);

		while(in.hasNext()) {
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
	
	private void generateAlgebraFeatureDataset() throws IOException {
		//this.trainingQueries = SparqlUtils.getQueries(inputQueryFile, prop.getProperty("Namespaces"), new ArrayList<>(), this.idColumn, this.queryColumn,this.execTimeColumn,this.input_delimiter.toCharArray()[0],"./unprocessed.txt");
		//QueryReaderLegacy reader = new QueryReaderLegacy(inputQueryFile);

		//this.trainingQueries = reader.getQueryData();

		//this.trainingQueries = SparqlUtils.getQueries(inputQueryFile, prop.getProperty("Namespaces"), new ArrayList<>(), this.idColumn, this.queryColumn,this.execTimeColumn,this.input_delimiter.toCharArray()[0],"./unprocessed.txt");
		QueryReaderWAlgRunTime.newLSQdata = true;
		QueryReaderWAlgRunTime.parseQueryString = false;
		QueryReaderWAlgRunTime.extra=true;
		ArrayList<lsq_extract.util.Query> qs= new ArrayList<>();
		/*lsq_extract.util.QueryReaderWAlgRunTime reader = new lsq_extract.util.QueryReaderWAlgRunTime(
				params[0]);*/
		lsq_extract.util.QueryReaderWAlgRunTime reader = new lsq_extract.util.QueryReaderWAlgRunTime(
				inputQueryFile);
		try {
			qs = reader.readFile();
		} catch (IOException e) {
			System.out.println("Something went wrong with the file path!");
		}
		List<String[]> queryStrs = new ArrayList<>();
		for(lsq_extract.util.Query q: qs){
			String[] record =new String[]{q.id,q.text, Double.toString(q.duration)};
			queryStrs.add(record);
		}
		reader.close();

		String[] header = new String[]{
				"triple", "bgp", "join", "leftjoin", "union", "filter", "graph", "extend", "minus", "path*",
				"pathN*", "path+", "pathN+", "path?", "notoneof", "tolist", "order", "project", "distinct", "reduced",
				"multi", "top", "group", "assign", "sequence", "slice", "treesize"};
		generateAlgebraFeatures(this.outputFile, header, queryStrs);
		System.out.println("Precess finished.");
	}
	
	private void generateAlgebraFeatures(String output, String[] header, List<String[]> queries) throws IOException {
		AlgebraFeatureExtractor fe = new AlgebraFeatureExtractor(header);
		BufferedWriter br;
		StringBuilder sb = new StringBuilder();
		try {
			br = new BufferedWriter(new FileWriter(output));
			{
				sb.append("queryID");
				sb.append(this.output_delimiter);
				int i = 0;
				while (i < header.length) {
					sb.append(header[i]);
					sb.append(this.output_delimiter);
					i++;
				}
				sb.append("execTime");
				sb.append(this.output_delimiter);
				sb.append("\n");
			}
			for(String[] q:queries) {
				try {
					double[] features = fe.extractFeatures(q[1]);
					//Print Id of query.
					sb.append(q[0]);
					sb.append(this.output_delimiter);
					for (double feature : features) {
						sb.append(feature);
						sb.append(this.output_delimiter);
					}
					sb.append(q[2]);
					sb.append(this.output_delimiter);
				}
				catch (Exception ex){
					ex.printStackTrace();
				}
				sb.append("\n");
			}
			br.write(sb.toString());
			br.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public static void produceALgebraFeatures(String inputFile, String outputFile, String prefixFile, String input_delimiter, String output_delimiter, int idColumn, int queryColumn, int execTimeColumn)  throws Exception{
		System.out.println("Inside algebra features generation");
		StopWatch watch = new StopWatch();
		watch.start();
		TDBExecutionAndFeature wrapper = new TDBExecutionAndFeature(inputFile, outputFile, prefixFile, input_delimiter, output_delimiter, idColumn, queryColumn, execTimeColumn);

		wrapper.generateAlgebraFeatureDataset();
		watch.stop();
		System.out.println("Total time for algebra query extraction: "+watch.getTime()+" ms");
	}

	public static void main(String[] args) throws Exception {
		System.out.println("Inside algebra features generation");
		String config_file = "";
		try{
			config_file = args[0];
		}
		catch (Exception ex) {
			System.out.println("You need to specify a config file url as first parameter");
			return;
		}
		StopWatch watch = new StopWatch();
		watch.start();		
		TDBExecutionAndFeature wrapper = new TDBExecutionAndFeature(config_file);

		wrapper.generateAlgebraFeatureDataset();
		watch.stop();
		System.out.println("Total time for algebra query extraction: "+watch.getTime()+" ms");
		
	}
	
}

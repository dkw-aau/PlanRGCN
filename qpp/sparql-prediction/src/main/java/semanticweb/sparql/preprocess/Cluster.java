package semanticweb.sparql.preprocess;

//import ged.AlgorithmConfig;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Scanner;
import java.util.concurrent.TimeUnit;

import Benchmark.Query;
import com.google.common.base.Stopwatch;

import ged.RDFGraphMatching;
//import org.apache.jena.ext.com.google.common.base.Stopwatch;
//import org.apache.jena.ext.com.google.common.base.Stopwatch;
//import org.apache.jena.ext.com.google.common.base.Stopwatch;
import semanticweb.sparql.config.ProjectConfiguration;
import semanticweb.sparql.utils.DBPediaUtils;

public class Cluster {
	private String centerFile = null;
	private String clusterFile = null;
	
	
	private String validationPredictionFile = "validation_cluster_prediction.dat";
	private String testPredictionFile = "test_cluster_prediction.dat";
	
	private String trainingFeatureExtraFile = null;
	private String validationFeatureExtraFile = null;
	private String testFeatureExtraFile = null;
	
	private String trainingSimilarityVectorfeature = null;
	private String validationSimilarityVectorFeature = null;
	private String testSimilarityVectorFeature = null;
	private String distanceMatrixFile = null;
	
	private List<Integer> center_idx = new ArrayList<Integer>();
	private List<Integer> idx = new ArrayList<Integer>();
	private List<Integer> validationIdx = new ArrayList<Integer>();
	private List<Integer> testIdx = new ArrayList<Integer>();
	
	private List<String> queries;
	private Properties prop;
	private float[][] distanceMatrix;
	private int K;
	
	public Cluster() throws IOException {
		prop = new Properties();
		prop.load(new FileInputStream(ProjectConfiguration.CONFIG_FILE));
		loadConfig();
		loadQuries();
		loadCenters();
		loadClusters();
		loadDistanceMatrix();
		
	}
	public void loadConfig() {
		centerFile = prop.getProperty("HungarianCenterCach");
		clusterFile = prop.getProperty("HungarianClusterCach");
		
		trainingFeatureExtraFile = prop.getProperty("TrainingExtraFeatures");
		validationFeatureExtraFile = prop.getProperty("ValidationExtraFeatures");
		testFeatureExtraFile = prop.getProperty("TestExtraFeatures");		
		
		trainingSimilarityVectorfeature = prop.getProperty("TrainingSimilarityVectorfeature");
		validationSimilarityVectorFeature = prop.getProperty("ValidationSimilarityVectorFeature");
		testSimilarityVectorFeature = prop.getProperty("TestSimilarityVectorFeature");
		distanceMatrixFile = prop.getProperty("TrainingDistanceHungarianMatrix");
	}
	
	public void loadQuries() throws IOException {
		String queryFile = prop.getProperty("TrainingQuery");
		/*FileInputStream fis = new FileInputStream(queryFile);
		Scanner in = new Scanner(fis);
		queries = null;
		queries = new ArrayList<String>();
		
		while(in.hasNext()) {
			String line = in.nextLine();
			//System.out.println(line);
			queries.add(line);
		}	
		fis.close();*/
		ArrayList<String> data = queryLoadHelper(queryFile);

		
	}

	public ArrayList<String> queryLoadHelper(String filepath){
		Scanner sc = null;
		File f = new File(filepath);
		ArrayList<String> data = new ArrayList<>();
		try {
			sc = new Scanner(f);
		} catch (FileNotFoundException e) {
			throw new RuntimeException(e);

		}
		int iteration = 0;
		while (sc.hasNextLine())
		{
			Scanner linescanner = new Scanner(sc.nextLine());
			linescanner.useDelimiter(",");
			List<String> lineData = new ArrayList<String>();
			linescanner.forEachRemaining(lineData::add);
			if (iteration > 0) {
				Query q = new Query();
				/*q.setId(lineData.get(0));
				q.setQueryString(lineData.get(1));
				q.setType(lineData.get(2));
				q.setNoTrplPatterns(Integer.parseInt( lineData.get(6)));
				/q.setRunTimeMS(Double.parseDouble(lineData.get(7)));
				q.setResultSize(Integer.parseInt( lineData.get(8)));*/
				data.add(lineData.get(1));
				linescanner.close();
			}
			iteration++;
		}
		sc.close();
		return data;
	}
	
	public void loadCenters() throws IOException {
		Scanner in = new Scanner(new FileInputStream(centerFile));
		while(in.hasNext()) {
			int id = in.nextInt();
			center_idx.add(id);
		}
		K = center_idx.size();
	}
	
	public void loadClusters() throws IOException {
		Scanner in = new Scanner(new FileInputStream(clusterFile));
		while(in.hasNext()) {
			int id = in.nextInt();
			idx.add(id);
		}
	}
	
	private class SimVec {
		double[] similarityVector = new double[K];
		int clusterId;
	}
	
	public SimVec predictCluster(String sparql) throws Exception{
		String q1 = DBPediaUtils.getQueryForDBpedia(sparql);
		
		SimVec sv = new SimVec();
		
		double minCost = Double.POSITIVE_INFINITY;
		int minCi = -1;
		for(int i=0;i<center_idx.size();i++) {
			int ci = center_idx.get(i);
			String sparql2 = queries.get(ci);
			String q2 = DBPediaUtils.getQueryForDBpedia(sparql2);
			//AlgorithmConfig algorithmConfig = AlgorithmConfig.createBipartiteHungarian();
			RDFGraphMatching matcher = new RDFGraphMatching();
			//System.out.println("q1: "+sparql1);
			//System.out.println("q2: "+sparql2);
			
			double cost = matcher.queryGraphDistance(q1, q2);
			sv.similarityVector[i] = 1.0/(1.0+cost);
			
			if(minCost>cost) {
				minCost = cost;
				minCi = ci;
			}
		}
		sv.clusterId = minCi;
		return sv;
	}
	
	public void processValidationQueries() throws Exception{
		Scanner in = new Scanner(new FileInputStream(prop.getProperty("ValidationQuery")));
		PrintStream ps = new PrintStream(validationPredictionFile);
		PrintStream psSimVec = new PrintStream(validationSimilarityVectorFeature);
		
		psSimVec.println(ProjectConfiguration.getPatternClusterSimVecFeatureHeader(K));
		
		System.out.println("Predicting clusters for validation dataset");
		Stopwatch watch = Stopwatch.createStarted();
		watch.start();		
		while(in.hasNext()) {
			String line = in.nextLine();
			SimVec sv = predictCluster(line);
			int c = sv.clusterId;
			validationIdx.add(c);
			//System.out.println(c);
			ps.println(c);
			writeSimFeatureVector(psSimVec, sv);
		}
		watch.stop();
		System.out.println("Elapsed time: "+watch.elapsed(TimeUnit.SECONDS)+" seconds");
		ps.close();
		psSimVec.close();
	}
	
	public void processTestQueries() throws Exception{
		
		//psTraining.println(ProjectConfiguration.getAlgebraFeatureHeader());
		//psValidation.println(ProjectConfiguration.getAlgebraFeatureHeader());
		//psTest.println(ProjectConfiguration.getAlgebraFeatureHeader());
		
		Scanner in = new Scanner(new FileInputStream(prop.getProperty("TestQuery")));
		PrintStream ps = new PrintStream(testPredictionFile);
		PrintStream psSimVec = new PrintStream(testSimilarityVectorFeature);
		psSimVec.println(ProjectConfiguration.getPatternClusterSimVecFeatureHeader(K));
		
		System.out.println("Predicting clusters for test dataset");
		Stopwatch watch = Stopwatch.createStarted();
		watch.start();		
		while(in.hasNext()) {
			String line = in.nextLine();
			SimVec sv = predictCluster(line);
			int c = sv.clusterId;
			testIdx.add(c);
			//System.out.println(c);
			ps.println(c);
			writeSimFeatureVector(psSimVec, sv);
		}
		watch.stop();
		System.out.println("Elapsed time: "+watch.elapsed(TimeUnit.SECONDS)+" seconds");
		ps.close();
		psSimVec.close();
	}
	
	private void writeSimFeatureVector(PrintStream ps, SimVec sv) {
		
		String outString = "";
		for(int j=0;j<sv.similarityVector.length;j++){
			if(j!=0) {
				outString += ",";
			}
			outString += sv.similarityVector[j];
		}
		ps.println(outString);
		
	}

	private void writeFreatureVectors(Map<Integer,Integer> cenIndex,List<Integer> list, PrintStream ps){
		int dimentions = K;
		for(int c:list) {
			int[] ftr = new int[dimentions];
			//System.out.println(c);
			//System.out.println(cenIndex.get(c));
			ftr[cenIndex.get(c)] = 1;
			
			String outString = "";
			for(int j=0;j<ftr.length;j++){
				if(j!=0) {
					outString += ",";
				}
				outString += ftr[j];
			}
			
			ps.println(outString);
			//System.out.println(outString);
		}		
	}
	
	public void createFeatures() throws IOException {
		System.out.println("Creating feature files for training, validation and test datasets");
		Stopwatch watch = Stopwatch.createStarted();
		watch.start();	
		
		PrintStream xFeaturePs = new PrintStream(trainingFeatureExtraFile);
		PrintStream xvalFeaturePs = new PrintStream(validationFeatureExtraFile);
		PrintStream xtestFeaturePs = new PrintStream(testFeatureExtraFile);
		
		
		
		Map<Integer,Integer> cenIndex = new HashMap<Integer, Integer>();
		int i=0;
		for(int c:center_idx){
			cenIndex.put(c, i);
			//System.out.println(c+" "+i);
			i++;
		}
		
		writeFreatureVectors(cenIndex, idx, xFeaturePs);
		writeFreatureVectors(cenIndex, validationIdx, xvalFeaturePs);
		writeFreatureVectors(cenIndex, testIdx, xtestFeaturePs);
		
		
		
		xFeaturePs.close();
		xvalFeaturePs.close();
		xtestFeaturePs.close();
		

		createTrainingSimilarityVectorFeatures();
		
		
		watch.stop();
		System.out.println("Elapsed time: "+watch.elapsed(TimeUnit.SECONDS)+" seconds");		
	}
	
	public void loadDistanceMatrix() throws IOException {
		System.out.println("Loading distance matrix");
		distanceMatrix = new float[queries.size()][queries.size()];
		
		Scanner in = new Scanner(new FileInputStream(distanceMatrixFile));
		int count = 0;
		while(in.hasNext()) {
			int i = in.nextInt();
			int j = in.nextInt();
			float d = in.nextFloat();
			distanceMatrix[i][j] = distanceMatrix[j][i] = d;
			
//			if(count%10000000==0) {
//				System.out.println(i+" "+j+" "+d);
//			}
			count++;
		}
		System.out.println("Loaded distance pairs:"+count);
	}
	
	public void createTrainingSimilarityVectorFeatures() throws IOException {
		PrintStream psSimVec = new PrintStream(trainingSimilarityVectorfeature);
		psSimVec.println(ProjectConfiguration.getPatternClusterSimVecFeatureHeader(K));
		for(int i=0;i<queries.size();i++) {
			int count=0;
			String outString = "";
			for(int ci:center_idx) {
				
				if(count!=0) {
					outString += ",";
				}
				outString += (1.0/(1.0+distanceMatrix[i][ci]));
				count++;
			}
			psSimVec.println(outString);
		}
		psSimVec.close();
	}
	public static void test() throws Exception{
		Cluster cl = new Cluster();
		cl.processValidationQueries();
		cl.processTestQueries();

		cl.createFeatures();
	}
	public static void main(String[] args) throws Exception {
		test();
		//cl.createTrainingSimilarityVectorFeatures();

		
	}

}

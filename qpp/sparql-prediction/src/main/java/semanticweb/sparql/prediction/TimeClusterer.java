package semanticweb.sparql.prediction;

//import ged.AlgorithmConfig;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintStream;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.List;
import java.util.Properties;
import java.util.Random;
import java.util.Scanner;
import java.util.concurrent.TimeUnit;

import Benchmark.Query;
import com.google.common.base.Stopwatch;
import com.google.common.primitives.Ints;

import ged.RDFGraphMatching;
//import org.apache.jena.ext.com.google.common.base.Stopwatch;
//import org.apache.jena.ext.com.google.common.base.Stopwatch;
//import org.apache.jena.ext.com.google.common.base.Stopwatch;
//import org.apache.jena.ext.com.google.common.base.Stopwatch;
import semanticweb.sparql.config.ProjectConfiguration;
import semanticweb.sparql.utils.DBPediaUtils;

import weka.clusterers.SimpleKMeans;
import weka.clusterers.XMeans;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.RemoveType;
import weka.filters.unsupervised.attribute.StringToNominal;

public class TimeClusterer {
	
	static int DEBUG = 0;
	final static int K_MEANS_MAX_ITERATION = 100;
	final static int K_MEANS_SEED = 100;
	
	private float[][] distanceMatrix;
	private int[] trainingClusterAssignments;
	private int[] mediodCenterIdx;

	private List<String> trainingQueries;
	private List<String> testQueries;
	private List<String> validationQueries;
	Instances exTimeInstances;
	
	private ProjectConfiguration config;
	int K;
	private int[] clusterSizes;
	
	private XMeans xmeans;
	
	
	public TimeClusterer() throws FileNotFoundException, IOException {
		config = new ProjectConfiguration();
		K = config.getNumberOfClusters();
	}
	public TimeClusterer(ProjectConfiguration i) throws FileNotFoundException, IOException {
		config = i;
		K = config.getNumberOfClusters();
	}
	
	Instances loadCSVNoHeaderOneAttribute(String filePath) throws IOException {
		
		CSVLoader loader = new CSVLoader();
		
		loader.setFile(new File( filePath));
		
		Instances instances = loader.getDataSet();
		
		
		
		
		//hack because the training csv does not have a header and weka assumes the first row will have headers
		
		//FastVector fv = new FastVector();
		
		FastVector atts = new FastVector();
		
		//atts.addElement(new Attribute(instances.attribute(0).name()));
		atts.addElement(instances.attribute(0));
		
		Instances newInstances = new Instances("Training", atts, 0);
		Instance instance = new DenseInstance(1);
		instance.setValue(0, Double.valueOf(instances.attribute(0).name()));
		newInstances.add(instance);
		//System.out.println(newInstances);
		//instances.add(instance);
		//Instances newInstances = Instances.mergeInstances(firstRow, instances);
		//instances.addAll(firstRow);
		
		Enumeration<Instance> e = instances.enumerateInstances();
		while(e.hasMoreElements()) {
			Instance ins = e.nextElement();
			//System.out.println(ins);
			newInstances.add(ins);
		}
		
		
		return newInstances;
	}
	

	public void clusterTrainingExecutionTime(int numberOfClusters) throws Exception {
		

		exTimeInstances = loadCSVNoHeaderOneAttribute(config.getTrainingQueryExecutionTimesFile());

		System.out.println("Instances: "+ exTimeInstances.numInstances());
		
		SimpleKMeans kmeans = new SimpleKMeans();
		kmeans.setSeed(K_MEANS_SEED);
		kmeans.setPreserveInstancesOrder(true);
		kmeans.setNumClusters(numberOfClusters);
		kmeans.setMaxIterations(K_MEANS_MAX_ITERATION);
		
		kmeans.buildClusterer(exTimeInstances);
		
		// This array returns the cluster number (starting with 0) for each instance
		// The array has as many elements as the number of instances
		
		trainingClusterAssignments = kmeans.getAssignments();
		mediodCenterIdx = new int[kmeans.numberOfClusters()];


		
		
		
		int k = 0;
		for(double num : kmeans.getClusterSizes()) {

			System.out.println("\nCluster: "+k+" Size:"+num);

			int i =0;
			Enumeration<Instance> e = exTimeInstances.enumerateInstances();
			List<Integer> clusterQuerieIndx = new ArrayList<Integer>();
			int ci = 0;
			while(e.hasMoreElements()) {
				Instance ins = e.nextElement();
				//System.out.println(ins);
				if(trainingClusterAssignments[i]==k) {
					double v = ins.value(0);
					
					clusterQuerieIndx.add(i);
					
					System.out.print(" ["+i+"]="+v);
				}
				
				i++;
			}
			
			double min_avg_distance = Double.POSITIVE_INFINITY;
			int min_c = -1;
			
			
			for(int qci:clusterQuerieIndx) {
				
				double d_sum = 0.0;
				for (int qcy:clusterQuerieIndx) {
					d_sum += distanceMatrix[qci][qcy];
				}
				
				double avg_distance = d_sum / clusterQuerieIndx.size();
				if(min_avg_distance> avg_distance) {
					min_avg_distance = avg_distance;
					min_c = qci;
				}
			}
			
			mediodCenterIdx[k] = min_c;

			
			
			
			System.out.println("\nMediod center time: "+ exTimeInstances.instance(mediodCenterIdx[k]));
			System.out.println("Mediod center id: "+ mediodCenterIdx[k]);
			
			k++;
		}
		
		//return mediodCenterIdx;
		
	}
	
	private class SimVec {
		//double[] similarityVector = new double[K];
		double[] classVector = new double[K];
		int clusterMediodId;
		int clusterIndex;
	}	
	
	public SimVec classifyQuery(String sparql1) throws Exception {
		SimVec sv = new SimVec();
		
		double minCost = Double.POSITIVE_INFINITY;
		int minCi = -1;
		int minI = -1;
		for(int i=0;i<mediodCenterIdx.length;i++) {
			int ci = mediodCenterIdx[i];
			String sparql2 = trainingQueries.get(ci);
			double cost = hungerianDistance(sparql1, sparql2);
			//sv.similarityVector[i] = 1.0/(1.0+cost);
			//sv.classVector[i] = 0;
			if(minCost>cost) {
				minCost = cost;
				minCi = ci;
				minI = i;
			}			
			
		}
		
		
		sv.clusterMediodId = minCi;
		sv.classVector[minI] = 1.0;
		sv.clusterIndex = minI;
		return sv;
	}
	
	private double hungerianDistance(String sparql1, String sparql2) throws Exception {
		//String q1 = DBPediaUtils.getQueryForDBpedia(sparql1); THESE ARE NOT RELEVANT FOR OUR (LSQ) DATASET
		//String q2 = DBPediaUtils.getQueryForDBpedia(sparql2); THESE ARE NOT RELEVANT FOR OUR (LSQ) DATASET
		//AlgorithmConfig algorithmConfig = AlgorithmConfig.createBipartiteHungarian();
		RDFGraphMatching matcher = new RDFGraphMatching();
		return matcher.queryGraphDistance(sparql1, sparql2);
	}
	/*
	public void createSimilarityVectorFeaturesKmeans(int [] center_idx, String outFile) throws IOException {
		PrintStream psSimVec = new PrintStream(outFile);
		DecimalFormat df=new DecimalFormat("0.000");
		for(int i=0;i<trainingQueries.size();i++) {
			int count=0;
			String outString = "";
			for(int ci:center_idx) {
				
				if(count!=0) {
					outString += ",";
				}
				outString += df.format((1.0/(1.0+distanceMatrix[i][ci])));
				count++;
			}
			psSimVec.println(outString);
		}
		psSimVec.close();
	}
	*/
	
	public void createClassVectorFeatures(String outFile) throws IOException {
		PrintStream ps = new PrintStream(outFile);
		
		ps.println(ProjectConfiguration.getTimeClusterBinaryVecFeatureHeader(K));
		
		DecimalFormat df=new DecimalFormat("0.000");
		for(int i=0;i<trainingQueries.size();i++) {
			int count=0;
			String outString = "";
			
			double[] classVector = new double[K];
			
			
			int classId = trainingClusterAssignments[i];
			classVector[classId] = 1.0;
			
			for(double c:classVector) {
				
				if(count!=0) {
					outString += ",";
				}
				
				outString += df.format(c);
				count++;
			}
			ps.println(outString);
		}
		ps.close();
	}	
	
	public void createTimeClassLabelFile(PrintStream ps, int[] assignments) throws IOException {
		//PrintStream ps = new PrintStream(config.getTrainingTimeClassKmeansFile());
		ps.println(ProjectConfiguration.getTimClusterLabelHeader());
		for(int label:assignments) {
			ps.println(label);
		}
		
		ps.close();
	}
	
	public void processTrainingDataKmeans() throws Exception {
		
		System.out.println("Clustering training dataset");
		Stopwatch watch = Stopwatch.createStarted();
		//watch.start();
		//int K = config.getNumberOfClusters();
		loadTrainingQuries();
		loadDistanceMatrix();
		
		clusterTrainingExecutionTime(K);
		
		createClassVectorFeatures(config.getTrainingClassFeatureKmeansFile());
		
		createTimeClassLabelFile(new PrintStream(config.getTrainingTimeClassKmeansFile()), trainingClusterAssignments);
		watch.stop();
		System.out.println("Elapsed time: "+watch.elapsed(TimeUnit.SECONDS)+" seconds");		
	}
	
	public void processTestData() throws Exception{
		
		
		List<Integer> assigments = new ArrayList<Integer>();
		
		testQueries = new ArrayList<String>();
		Scanner in = new Scanner(new FileInputStream(config.getTestQueryFile()));

		PrintStream psClassVec = new PrintStream(config.getTestClassVectorFeatureKmeansFile());
		psClassVec.println(ProjectConfiguration.getTimeClusterBinaryVecFeatureHeader(K));
		
		
		System.out.println("Predicting clusters for test dataset");
		Stopwatch watch = Stopwatch.createStarted();
		//watch.start();
		int i = 0;
		while(in.hasNext()) {
			String line = in.nextLine();
			SimVec sv = classifyQuery(line);
			int c = sv.clusterMediodId;
			//writeSimFeatureKmeansVector(psSimVec, sv);
			writeClassFeatureKmeansVector(psClassVec,sv);
			
			testQueries.add(line);
			
			//assignmentTestQuery[i] = sv.clusterIndex;
			assigments.add(sv.clusterIndex);
			i++;
			
		}
		int[] assignmentTestQuery = Ints.toArray(assigments);
		createTimeClassLabelFile(new PrintStream(config.getTestTimeClassKmeansFile()), assignmentTestQuery);
		
		watch.stop();
		System.out.println("Elapsed time: "+watch.elapsed(TimeUnit.SECONDS)+" seconds");
		//ps.close();
		//psSimVec.close();
		psClassVec.close();
		in.close();
		
		
		
		//for debug
		if(DEBUG==0) return;
		
		
		Scanner inY = new Scanner(new FileInputStream(config.getTestQueryExecutionTimesFile()));
		
		List<Double> yTest = new ArrayList<Double>();
		while(inY.hasNext()) {
			double t = (double) inY.nextInt();
			yTest.add(t);
		}
		inY.close();
		System.out.println(Arrays.toString(assignmentTestQuery));
		//System.out.println();
		
		int k=0;
		for(int md:mediodCenterIdx) {
			
			System.out.println("\nCluster: "+k);
			
			//List<Integer> clusterIdx = new ArrayList<Integer>();
			int num = 0;
			for(int x=0;x<assignmentTestQuery.length;x++) {
				//System.out.print(" "+x+" "+assignmentTestQuery[x]);
				if(assignmentTestQuery[x] == k) {
					//clusterIdx.add(x);
					System.out.print(" ["+x+"]="+yTest.get(x));
					num++;
					
				}
			}
			
			System.out.println("\nSize:"+num);
			
			System.out.println("\nMediod center time: "+ exTimeInstances.instance(mediodCenterIdx[k]));
			System.out.println("Mediod center id: "+ md);
			k++;
			//break;
		}

	}	
	
	public void processValidationData() throws Exception{
		List<Integer> assigments = new ArrayList<Integer>();
		validationQueries = new ArrayList<String>();
		Scanner in = new Scanner(new FileInputStream(config.getValidationQueryFile()));
		//PrintStream ps = new PrintStream(validationPredictionFile);
		//PrintStream psSimVec = new PrintStream(config.getValidationSimilarityVectorFeatureKmeansFile());
		PrintStream psClassVec = new PrintStream(config.getValidationClassFeatureKmeansFile());
		psClassVec.println(ProjectConfiguration.getTimeClusterBinaryVecFeatureHeader(K));
		
		
		System.out.println("Predicting clusters for validation dataset");
		Stopwatch watch = Stopwatch.createStarted();
		//watch.start();
		while(in.hasNext()) {
			String line = in.nextLine();
			SimVec sv = classifyQuery(line);
			int c = sv.clusterMediodId;
			//writeSimFeatureKmeansVector(psSimVec, sv);
			writeClassFeatureKmeansVector(psClassVec,sv);
			
			validationQueries.add(line);
			assigments.add(sv.clusterIndex);
			
		}
		
		int[] assignmentValQuery = Ints.toArray(assigments);
		createTimeClassLabelFile(new PrintStream(config.getValidationTimeClassKmeansFile()), assignmentValQuery);
		
		watch.stop();
		System.out.println("Elapsed time: "+watch.elapsed(TimeUnit.SECONDS)+" seconds");
		//ps.close();
		psClassVec.close();
		//psSimVec.close();
		in.close();
	}
	
	
	private void writeClassFeatureKmeansVector(PrintStream ps, SimVec sv) {
		
		DecimalFormat df=new DecimalFormat("0.000");
		String outString = "";
		for(int j=0;j<sv.classVector.length;j++){
			if(j!=0) {
				outString += ",";
			}
			outString += df.format(sv.classVector[j]);
		}
		ps.println(outString);
		
	}
	/*
	private void writeSimFeatureKmeansVector(PrintStream ps, SimVec sv) {
		
		DecimalFormat df=new DecimalFormat("0.000");
		String outString = "";
		for(int j=0;j<sv.similarityVector.length;j++){
			if(j!=0) {
				outString += ",";
			}
			outString += df.format(sv.similarityVector[j]);
		}
		ps.println(outString);
		
	}		
	*/
	public void loadDistanceMatrix() throws IOException {
		System.out.println("Loading distance matrix");
		distanceMatrix = new float[trainingQueries.size()][trainingQueries.size()];
		
		Scanner in = new Scanner(new FileInputStream(config.getDistanceMatrixFile()));
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
	
	public void loadTrainingQuries() throws IOException {
		String queryFile = config.getTrainingQueryFile();
		//trainingQueries = queryLoadHelper(queryFile);
		FileInputStream fis = new FileInputStream(queryFile);
		Scanner in = new Scanner(fis);
		trainingQueries = null;
		trainingQueries = new ArrayList<String>();
		
		while(in.hasNext()) {
			String line = in.nextLine();
			//System.out.println(line);
			trainingQueries.add(line);
		}	
		fis.close();
		
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
	public void createKmeansClusterFeatures() throws Exception {
		processTrainingDataKmeans();
		processValidationData();
		processTestData();
	}
	
	public void debug() throws Exception {
		
		/*
		processTrainingData();
		
		testQueries = new ArrayList<String>();
		Scanner in = new Scanner(new FileInputStream(config.getTestQueryFile()));
		//PrintStream ps = new PrintStream(validationPredictionFile);
		PrintStream psSimVec = new PrintStream(config.getTestSimilarityVectorFeatureKmeansFile());
		
		Stopwatch watch = new Stopwatch();
		watch.start();		
		while(in.hasNext()) {
			String line = in.nextLine();
			//SimVec sv = classifyQuery(line);
			//int c = sv.clusterId;
			//writeSimFeatureVector(psSimVec, sv);
			
			testQueries.add(line);
			
		}	
		in.close();
		
		
		
		clusterTrainingExecutionTime(K);
		
		*/
		
		
		
		
		String q1 = testQueries.get(1180);
		System.out.println("Test query:");
		DBPediaUtils.prettyPrintSparql(q1);
		
		System.out.println(mediodCenterIdx.length+" mediod queries:");
		
		int k = 0;
		
		for(int ci:mediodCenterIdx) {
			String q2 = trainingQueries.get(ci);
			System.out.println("Cluster:"+k);
			System.out.println("Mediod:"+ci);
			DBPediaUtils.prettyPrintSparql(q2);
			double d = hungerianDistance(q1, q2);
			System.out.println("hungarian:"+d);
			
			k++;
			System.out.println("------------------------------------------------------------------");
			
		}
		SimVec sv = classifyQuery(q1);
		
		//System.out.println(Arrays.toString(sv.similarityVector));
		System.out.println(Arrays.toString(sv.classVector));
		
		
	}
	private void debug1() throws IOException {
		
		Instances exTimeInstances = loadCSVNoHeaderOneAttribute(config.getTrainingQueryExecutionTimesFile());
	}
	
	public void setDebugMode(){
		DEBUG = 1;
	}
	
	public void clusterXmeansTrainingExecutionTime() throws Exception {
		
		
		K = config.getxMeansMaxK();
		CSVLoader loader = new CSVLoader();
		
		loader.setFile(new File( config.getTrainingQueryExecutionTimesFile()));
		
		exTimeInstances = loader.getDataSet();

		

		System.out.println("Instances: "+ exTimeInstances.numInstances());
		
		xmeans = new XMeans();
		xmeans.setSeed(K_MEANS_SEED);
		xmeans.setMaxIterations(100);
		xmeans.setCutOffFactor(0.5);
		xmeans.setMaxKMeans(1000);
		xmeans.setMaxKMeansForChildren(1000);
		
		xmeans.setMaxNumClusters(K);
		xmeans.setMinNumClusters(2);
		
		xmeans.buildClusterer(exTimeInstances);
		K = xmeans.numberOfClusters();
		//xmeans.get
		
		trainingClusterAssignments = new int[exTimeInstances.numInstances()];
		clusterSizes = new int[K];
		
		for(int i=0;i<exTimeInstances.numInstances();i++) {
			Instance ins = exTimeInstances.instance(i);
			trainingClusterAssignments[i] = xmeans.clusterInstance(ins);
			clusterSizes[trainingClusterAssignments[i]]++;
		}
		
		

//		
//		// This array returns the cluster number (starting with 0) for each instance
//		// The array has as many elements as the number of instances
//		
//		trainingKmeansAssignments = kmeans.getAssignments();
//		mediodCenterIdx = new int[kmeans.numberOfClusters()];
//

		
		
		
		int k = 0;
		for(int num:clusterSizes) {
			System.out.println("\nCluster: "+k+" Size:"+num);

			int i =0;
			Enumeration<Instance> e = exTimeInstances.enumerateInstances();
			List<Integer> clusterQuerieIndx = new ArrayList<Integer>();
			int ci = 0;
			while(e.hasMoreElements()) {
				Instance ins = e.nextElement();
				//System.out.println(ins);
				if(trainingClusterAssignments[i]==k) {
					double v = ins.value(0);
					
					clusterQuerieIndx.add(i);
					
					System.out.print(" ["+i+"]="+v);
				}
				
				i++;
			}
			
			
			//System.out.println("\nMediod center time: "+ exTimeInstances.instance(mediodCenterIdx[k]));
			//System.out.println("Mediod center id: "+ mediodCenterIdx[k]);
			System.out.println("\nCentroid:"+xmeans.getClusterCenters().instance(k));
			
			k++;
		}
//		
//		//return mediodCenterIdx;
		
	}
		
	public void generateValidationLabelDataXmeans() throws Exception {
		
		//PrintStream ps = new PrintStream(new File(config.getValidationTimeClassXmeansFile()));
		//ps.println(ClusteringConfiguration.getTimClusterLabelHeader());
		
		CSVLoader valLoader = new CSVLoader();
		valLoader.setFile(new File(config.getValidationQueryExecutionTimesFile()));
		
		Instances valInstances = valLoader.getDataSet();
		
		Enumeration<Instance> e = valInstances.enumerateInstances();
		
		List<Integer> asnmnts = new ArrayList<Integer>();
		while(e.hasMoreElements()) {
			Instance ins = e.nextElement();
			int c = xmeans.clusterInstance(ins); 
			asnmnts.add(c);
			
		}
		createTimeClassLabelFile(new PrintStream(new File(config.getValidationTimeClassXmeansFile())), Ints.toArray(asnmnts));
		//ps.close();
		
		
	}
	public void generateTestLabelDataXmeans() throws Exception {
		
		//PrintStream ps = new PrintStream(new File(config.getValidationTimeClassXmeansFile()));
		//ps.println(ClusteringConfiguration.getTimClusterLabelHeader());
		
		CSVLoader testLoader = new CSVLoader();
		testLoader.setFile(new File(config.getTestQueryExecutionTimesFile()));
		
		Instances testInstances = testLoader.getDataSet();
		
		Enumeration<Instance> e = testInstances.enumerateInstances();
		
		List<Integer> asnmnts = new ArrayList<Integer>();
		while(e.hasMoreElements()) {
			Instance ins = e.nextElement();
			int c = xmeans.clusterInstance(ins); 
			asnmnts.add(c);
			
		}
		createTimeClassLabelFile(new PrintStream(new File(config.getTestTimeClassXmeansFile())), Ints.toArray(asnmnts));
		//ps.close();
		
		
	}	
	
	public void processValidationDataXMeans() throws Exception {
		Scanner in = new Scanner(new FileInputStream(config.getValidationQueryFile()));
		testQueries = new ArrayList<String>();
		while(in.hasNext()) {
			String query = in.nextLine();
			testQueries.add(query);
		}
		in.close();
		
		generateValidationLabelDataXmeans();
	}
	
	public void processTestDataXmeans() throws Exception {
		Scanner in = new Scanner(new FileInputStream(config.getTestQueryFile()));
		validationQueries = new ArrayList<String>();
		while(in.hasNext()) {
			String query = in.nextLine();
			validationQueries.add(query);
		}
		in.close();
		
		generateTestLabelDataXmeans();
		
	}
	
	public void processTrainingDataXmeans() throws Exception {
		
		System.out.println("Clustering training dataset (Xmeans)");
		Stopwatch watch = Stopwatch.createStarted();
		//watch.start();
		//int K = config.getNumberOfClusters();
		loadTrainingQuries();
		//loadDistanceMatrix();
		
		clusterXmeansTrainingExecutionTime();
		
		//createClassVectorFeatures(config.getTrainingClassFeatureKmeansFile());
		createClassVectorFeatures(config.getTrainingClassFeatureXmeansFile());
		
		//createTimeClassLabelFile(new PrintStream(config.getTrainingTimeClassKmeansFile()), trainingKmeansAssignments);
		createTimeClassLabelFile(new PrintStream(config.getTrainingTimeClassXmeansFile()), trainingClusterAssignments);
		watch.stop();
		System.out.println("Elapsed time: "+watch.elapsed(TimeUnit.SECONDS)+" seconds");		
	}	
	
	public int[] getClusterAssignments() {
		return trainingClusterAssignments;
	}
	
	public int getNumberOfClusters() {
		return K;
	}

	public static void test() throws Exception{
		TimeClusterer km = new TimeClusterer();

		//km.setDebugMode();
		//km.createKmeansClusterFeatures();

		//km.debug();

		km.processTrainingDataXmeans();
		//km.processValidationDataXMeans();
		//km.processTestDataXmeans();
	}
	
	
	public static void main(String[] args) throws Exception {
		test();
	}


}

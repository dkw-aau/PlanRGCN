package semanticweb.sparql.suggestion;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.apache.commons.math3.random.RandomData;
import org.apache.commons.math3.random.RandomDataGenerator;

import semanticweb.sparql.config.ProjectConfiguration;
import semanticweb.sparql.utils.AttributeFilterMeta;
import semanticweb.sparql.utils.DBPediaUtils;
import semanticweb.sparql.utils.GeneralUtils;
import semanticweb.sparql.utils.WekaUtils;
import weka.core.Attribute;
import weka.core.DistanceFunction;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.neighboursearch.ModifiedKDTree;
import weka.core.neighboursearch.NearestNeighbourSearch;
import weka.core.neighboursearch.kdtrees.KDTreeNodeSplitter;
import weka.core.neighboursearch.kdtrees.SlidingMidPointOfWidestSide;

public class QuerySuggester {
	private static String CONFIG_FILE = System.getProperty("user.home")+"/Documents/code/query-performance/config-knn-6000.prop";
	private ProjectConfiguration config;
	private Instances trainingInstances;
	private Instances validationInstances;
	private Instances testInstances;
	private AttributeFilterMeta trainingFeaturesMeta;
	private List<String> trainingQueries;
	private List<String> validationQueries;
	private List<String> testQueries;
	private ModifiedKDTree kdTree;
	//private Map<Instance,Integer> trainingInstanceIndex;
	private Map<String,Integer> trainingInstanceIndex;
	
	public QuerySuggester() throws IOException {
		config = new ProjectConfiguration(CONFIG_FILE);
	}
	
	
	public void loadData() throws Exception {
		loadTrainingData();
		loadValidationtData();
		loadTestData();

	}
	
	public void loadTrainingData() throws Exception {
		Instances trainingAlgebraFeatureInstances = WekaUtils.loadCSV(config.getTrainingAlgebraFeaturesFile());
		Instances trainingSimVecFeatureInstances = WekaUtils.loadCSV(config.getTrainingSimilarityVectorfeatureFile());
		//Instances trainingExecutionTimeInstances = WekaUtils.loadCSV(config.getTrainingQueryExecutionTimesFile());
		
		trainingFeaturesMeta = WekaUtils.refineInstances(trainingAlgebraFeatureInstances, trainingSimVecFeatureInstances);
		
		Instances features = trainingFeaturesMeta.getInstances();
		//trainingInstances = WekaUtils.addNumericLablesForRegression(features, trainingFeaturesMeta, trainingExecutionTimeInstances);
		trainingInstances = features;
		loadTrainingQueries();
		//indexTrainingInstacesAndRemoveRepetations();
		System.out.println("Total unique training queries:"+trainingInstances.numInstances());
		WekaUtils.saveInstances(trainingInstances, "/Users/hrakebul/Documents/code/query-performance/6000-knn/x_features_suggestions.arff");
	}	
	



	public void loadValidationtData() throws Exception {
		Instances validationAlgebraFeatureInstances = WekaUtils.loadCSV(config.getValidationAlgebraFeaturesFile());
		Instances validationSimVecFeatureInstances = WekaUtils.loadCSV(config.getValidationSimilarityVectorFeatureFile());
		//Instances validationtExecutionTimeInstances = WekaUtils.loadCSV(config.getValidationQueryExecutionTimesFile());
		
		Instances features = WekaUtils.refineInstances(validationAlgebraFeatureInstances, validationSimVecFeatureInstances,trainingFeaturesMeta);
		//WekaUtils.refineInstances(algebraInstances, simVecInstances, meta)
		
		//Instances features = featuresMeta.getInstances();
		//validationInstances = WekaUtils.addNumericLablesForRegression(features, trainingFeaturesMeta, validationtExecutionTimeInstances);
		validationInstances = features;
		loadValidationQueries();
	}
	
	public void loadTestData() throws Exception {
		Instances testAlgebraFeatureInstances = WekaUtils.loadCSV(config.getTestAlgebraFeaturesFile());
		Instances testSimVecFeatureInstances = WekaUtils.loadCSV(config.getTestSimilarityVectorFeatureFile());
		//Instances testExecutionTimeInstances = WekaUtils.loadCSV(config.getTestQueryExecutionTimesFile());
		
		Instances features = WekaUtils.refineInstances(testAlgebraFeatureInstances, testSimVecFeatureInstances,trainingFeaturesMeta);
		//WekaUtils.refineInstances(algebraInstances, simVecInstances, meta)
		
		//Instances features = featuresMeta.getInstances();
		//testInstances = WekaUtils.addNumericLablesForRegression(features, trainingFeaturesMeta, testExecutionTimeInstances);
		testInstances = features;
		loadTestQueries();
		
		//System.out.println(testInstances);
	}
	
	public void loadTrainingQueries() throws IOException {
		String trainingQueryFile = config.getTrainingQueryFile();
		trainingQueries = GeneralUtils.loadQuries(trainingQueryFile);
	}

	public void loadValidationQueries() throws IOException {
		String validationQueryFile = config.getValidationQueryFile();
		validationQueries = GeneralUtils.loadQuries(validationQueryFile);
	}
	public void loadTestQueries() throws IOException {
		String testQueryFile = config.getTestQueryFile();
		testQueries = GeneralUtils.loadQuries(testQueryFile);
	}
	
	public void trainNearestNeighbourSearch() throws Exception {
		kdTree = new ModifiedKDTree();
		kdTree.setInstances(trainingInstances);
		//EuclideanDistance df = new EuclideanDistance();
		//df.setDontNormalize(true);
		//df.setAttributeIndices("first-last");
		//df.setDontNormalize(false);
		//df.setInvertSelection(false);
		//df.setInstances(trainingInstances);
		
		//kdTree.setDistanceFunction(df);
		
		
		//kdTree.setMaxInstInLeaf(40);
		//kdTree.setMeasurePerformance(false);
		//kdTree.setMinBoxRelWidth(0.01);
		//KDTreeNodeSplitter splitter = new SlidingMidPointOfWidestSide();
		
		
		//kdTree.setNodeSplitter(splitter);
		//kdTree.setNormalizeNodeWidth(true);
		
		
		
		
	}
	
	public void testWithRandomTestQuery() throws Exception{
		//int testQueryIndex = getRandomTestQueryIndex();
		int testQueryIndex = 1180;
		Instance testQueryInstance = testInstances.instance(testQueryIndex);
		String testQueryStringEncoded = testQueries.get(testQueryIndex);
		System.out.println("Generating suggestions for the query:");
		DBPediaUtils.prettyPrintSparql(testQueryStringEncoded);
		
		int[] suggestionIndices = getKSuggestions(testQueryInstance, 5);
		
		System.out.println("Query suggestions are:");
		System.out.println(Arrays.toString(suggestionIndices));
		for(int index:suggestionIndices) {
			System.out.println("--------------------");
			DBPediaUtils.prettyPrintSparql(trainingQueries.get(index));
		}
	}
	
	public void testWithRandomTrainingQuery() throws Exception{
		int testQueryIndex = getRandomTtrainingQueryIndex();
		Instance testQueryInstance = trainingInstances.instance(testQueryIndex);
		String testQueryStringEncoded = trainingQueries.get(testQueryIndex);
		System.out.println("Generating suggestions for the query:");
		DBPediaUtils.prettyPrintSparql(testQueryStringEncoded);
		
		int[] suggestionIndices = getKSuggestions(testQueryInstance, 3);
		
		System.out.println("Query suggestions are:");
		System.out.println(Arrays.toString(suggestionIndices));
		for(int index:suggestionIndices) {
			System.out.println("--------------------");
			DBPediaUtils.prettyPrintSparql(trainingQueries.get(index));
		}
	}	
	/*
	public int[] getKSuggestions(Instance target,int k) throws Exception {
		//System.out.println(kdTree);
		//System.out.println(target);
		Instances instances = kdTree.kNearestNeighbours(target, k);
		//System.out.println(instances);
		//System.out.println("Nearest:\n"+kdTree.nearestNeighbour(target));
		int nnumberOfInstances = instances.numInstances()>k?k:instances.numInstances();
		//System.out.println(instances.numInstances());
		int[] neighboursIndex = new int[nnumberOfInstances];
		Enumeration<Instance> en = instances.enumerateInstances();
		int i = 0;
		
		while(en.hasMoreElements() && i<nnumberOfInstances) {
			Instance instance = en.nextElement();
			int index = getIndex(instance);
			//int index = trainingInstances.indexOf(instance);
			neighboursIndex[i] = index;
			i++;
			
			//DBPediaUtils.prettyPrintSparql(trainingQueries.get(index));
		}
		
		return neighboursIndex;
	}*/
	public int[] getKSuggestions(Instance target,int k) throws Exception {
		//System.out.println(kdTree);
		//System.out.println(target);
		int[] computedNeighboursIndex = kdTree.kNearestNeighboursModified(target, k);
		//System.out.println(instances);
		//System.out.println("Nearest:\n"+kdTree.nearestNeighbour(target));
		int numberOfInstances = computedNeighboursIndex.length>k?k:computedNeighboursIndex.length;
		//System.out.println(instances.numInstances());
		int[] neighboursIndex = new int[numberOfInstances];
		for(int i=0;i<numberOfInstances;i++) {
			neighboursIndex[i] = computedNeighboursIndex[i];
		}
		
		return neighboursIndex;
	}	
	
	private void indexTrainingInstacesAndRemoveRepetations() {
		//trainingInstanceIndex = new HashMap<Instance, Integer>();
		trainingInstanceIndex = new HashMap<String, Integer>();
		Enumeration<Instance> en = trainingInstances.enumerateInstances();
		ArrayList<String> qList = new ArrayList<String>();
		ArrayList<Attribute> attInfo = new ArrayList<Attribute>();
		for(int x=0;x<trainingInstances.numAttributes();x++) {
			attInfo.add(trainingInstances.attribute(x));
		}
		
		
		Instances instances = new Instances("training", attInfo, 0);
		int i = 0;
		int newIndex = 0;
		while(en.hasMoreElements()) {
			Instance instance = en.nextElement();
			String key = instance.toString();
			if(trainingInstanceIndex.containsKey(key)==false) {
				trainingInstanceIndex.put(key, newIndex);
				instances.add(instance);
				qList.add(trainingQueries.get(i));
				newIndex++;
			}
			i++;
		}
		trainingQueries = qList;
		trainingInstances = instances;
	}
	private int getIndex(Instance instance) {
		String key = instance.toString();
		return trainingInstanceIndex.containsKey(key)==true?trainingInstanceIndex.get(key):-1;
	}
	private int getRandomTestQueryIndex() {
		RandomDataGenerator randomData = new RandomDataGenerator();
		return randomData.nextInt(0, testQueries.size()-1);
		
	}
	
	private int getRandomTtrainingQueryIndex() {
		RandomDataGenerator randomData = new RandomDataGenerator();
		return randomData.nextInt(0, trainingInstances.size()-1);
		
	}
	
	public static void main(String[] args) throws Exception {
		QuerySuggester qs = new QuerySuggester();
		qs.loadData();
		qs.trainNearestNeighbourSearch();
		qs.testWithRandomTestQuery();
		//qs.testWithRandomTrainingQuery();
	}
}

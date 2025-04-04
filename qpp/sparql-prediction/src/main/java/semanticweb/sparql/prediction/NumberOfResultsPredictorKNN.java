package semanticweb.sparql.prediction;

import java.io.IOException;
import java.util.HashMap;

import semanticweb.sparql.config.ProjectConfiguration;
import semanticweb.sparql.utils.AttributeFilterMeta;
import semanticweb.sparql.utils.GeneralUtils;
import semanticweb.sparql.utils.StatUtils;
import semanticweb.sparql.utils.WekaUtils;



import weka.classifiers.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.DistanceFunction;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.Tag;
import weka.core.neighboursearch.KDTree;
import weka.core.neighboursearch.NearestNeighbourSearch;

public class NumberOfResultsPredictorKNN {
	//public static String CONFIG_FILE = System.getProperty("user.home")+"/Documents/code/query-performance/config-knn-6000.prop";
	//public static String CONFIG_FILE = System.getProperty("user.home")+"/Documents/code/query-performance/config-knn-dbpsb-test.prop";
	
	//public static String CONFIG_FILE = System.getProperty("user.home")+"/Documents/code/query-performance/config-dbpsb-k5.prop";
	//public static String CONFIG_FILE = System.getProperty("user.home")+"/Documents/code/query-performance/config-dbpsb-k10.prop";
	//public static String CONFIG_FILE = System.getProperty("user.home")+"/Documents/code/query-performance/config-dbpsb-k15.prop";
	//public static String CONFIG_FILE = System.getProperty("user.home")+"/Documents/code/query-performance/config-dbpsb-k20.prop";
	public static String CONFIG_FILE = System.getProperty("user.home")+"/Documents/code/query-performance/config-dbpsb-k25.prop";
	
	
	ProjectConfiguration config;
	Instances trainingInstances;
	Instances validationInstances;
	Instances testInstances;
	AttributeFilterMeta trainingFeaturesMeta;
	IBk knnModel;
	int K4KNN = 5;
	
	public NumberOfResultsPredictorKNN() throws IOException {
		config = new ProjectConfiguration(CONFIG_FILE);
	}
	
	public void loadData() throws Exception {
		loadTrainingData();
		loadValidationtData();
		loadTestData();
		WekaUtils.saveInstances(trainingInstances, config.getTrainingARFFFile());
		WekaUtils.saveInstances(validationInstances, config.getValidationARFFFile());
		WekaUtils.saveInstances(testInstances, config.getTestARFFFile());
	}
	
	public void loadTrainingData() throws Exception {
		Instances trainingAlgebraFeatureInstances = WekaUtils.loadCSV(config.getTrainingAlgebraFeaturesFile());
		Instances trainingSimVecFeatureInstances = WekaUtils.loadCSV(config.getTrainingSimilarityVectorfeatureFile());
		Instances trainingNumOfResultsInstances = WekaUtils.loadCSV(config.getTrainingNumberOfRecordsFile());
		
		trainingFeaturesMeta = WekaUtils.refineInstances(trainingAlgebraFeatureInstances, trainingSimVecFeatureInstances);
		
		Instances features = trainingFeaturesMeta.getInstances();
		trainingInstances = WekaUtils.addNumericLablesForRegression(features, trainingFeaturesMeta, trainingNumOfResultsInstances); 
		//regressionTrainingDatasets = new HashMap<Integer, Instances>();
	}	
	
	public void loadValidationtData() throws Exception {
		Instances validationAlgebraFeatureInstances = WekaUtils.loadCSV(config.getValidationAlgebraFeaturesFile());
		Instances validationSimVecFeatureInstances = WekaUtils.loadCSV(config.getValidationSimilarityVectorFeatureFile());
		Instances validationtNumberOfRecordsInstances = WekaUtils.loadCSV(config.getValidationNumberOfRecordsFile());
		
		Instances features = WekaUtils.refineInstances(validationAlgebraFeatureInstances, validationSimVecFeatureInstances,trainingFeaturesMeta);
		//WekaUtils.refineInstances(algebraInstances, simVecInstances, meta)
		
		//Instances features = featuresMeta.getInstances();
		validationInstances = WekaUtils.addNumericLablesForRegression(features, trainingFeaturesMeta, validationtNumberOfRecordsInstances); 
		
	}
	
	public void loadTestData() throws Exception {
		Instances testAlgebraFeatureInstances = WekaUtils.loadCSV(config.getTestAlgebraFeaturesFile());
		Instances testSimVecFeatureInstances = WekaUtils.loadCSV(config.getTestSimilarityVectorFeatureFile());
		Instances testNumberOfRecordsInstances = WekaUtils.loadCSV(config.getTestNumberOfRecordsFile());
		
		Instances features = WekaUtils.refineInstances(testAlgebraFeatureInstances, testSimVecFeatureInstances,trainingFeaturesMeta);
		//WekaUtils.refineInstances(algebraInstances, simVecInstances, meta)
		
		//Instances features = featuresMeta.getInstances();
		testInstances = WekaUtils.addNumericLablesForRegression(features, trainingFeaturesMeta, testNumberOfRecordsInstances); 
		
	}
	
	
	
	public void trainKNNRegression() throws Exception {
		knnModel = new IBk();
		knnModel.setKNN(K4KNN);
		SelectedTag inverseDistance = new SelectedTag(IBk.WEIGHT_INVERSE, IBk.TAGS_WEIGHTING);
		
		
		knnModel.setDistanceWeighting(inverseDistance);
		
		NearestNeighbourSearch kdTree = new KDTree();
		DistanceFunction df = new EuclideanDistance();
        //df.setDontNormalize(true);
        //tree.setDistanceFunction(df); 
		
		kdTree.setDistanceFunction(df);
		
		
		knnModel.setNearestNeighbourSearchAlgorithm(kdTree);
		knnModel.buildClassifier(trainingInstances);
		
		Evaluation eval = new Evaluation(trainingInstances);
		double[] y = eval.evaluateModel(knnModel, trainingInstances);
		GeneralUtils.saveExecutionTimePredictions(y, config.getTrainingQueryExecutionTimesPredictedFile());
		
		System.out.println(eval.toSummaryString("\nTraining Results\n======\n", false));
		//System.out.println(eval.toMatrixString());
		System.out.println("R-squared (training): "+eval.correlationCoefficient()*eval.correlationCoefficient());

//		double[] predictedValues = new double[trainingInstances.numInstances()];
//		double[] originalValues = new double[trainingInstances.numInstances()];
//		
//		//int[] predictedClasses = new int[trainingInstances.numInstances()];
//		for(int i=0;i<trainingInstances.numInstances();i++) {
//			Instance instance = trainingInstances.instance(i);
//			
//			originalValues[i] = instance.classValue();
//			predictedValues[i] = y[i];
//			//predictedClasses[i] = timeClassifier.classifyInstance(instance); 
//		}
//		
//		
//		double rSquared = StatUtils.rSquared(originalValues, predictedValues);
//		System.out.println("R-squared (training): "+rSquared);		
		
	}
	
	
	public void crossValidation() throws Exception {
		Evaluation eval = new Evaluation(trainingInstances);
		double[] y = eval.evaluateModel(knnModel, validationInstances);
		GeneralUtils.saveExecutionTimePredictions(y, config.getValidationQueryExecutionTimesPredictedFile());
		System.out.println(eval.toSummaryString("\nValidation Results\n======\n", false));
		System.out.println("R-squared (validation): "+eval.correlationCoefficient()*eval.correlationCoefficient());
		
//		double[] predictedValues = new double[testInstances.numInstances()];
//		double[] originalValues = new double[testInstances.numInstances()];
//		for(int i=0;i<testInstances.numInstances();i++) {
//			Instance instance = testInstances.instance(i);
//			
//			originalValues[i] = instance.classValue();
//			predictedValues[i] = y[i];
//
//		}
//		
//		double rSquared = StatUtils.rSquared(originalValues, predictedValues);
//		System.out.println("R-squared (test): "+rSquared);
		
		
		
	}	
	public void testModel() throws Exception {
		Evaluation eval = new Evaluation(trainingInstances);
		double[] y = eval.evaluateModel(knnModel, testInstances);
		GeneralUtils.saveExecutionTimePredictions(y, config.getTestQueryExecutionTimesPredictedFile());
		System.out.println(eval.toSummaryString("\nTest Results\n======\n", false));
		System.out.println("R-squared (test): "+eval.correlationCoefficient()*eval.correlationCoefficient());
		
//		double[] predictedValues = new double[testInstances.numInstances()];
//		double[] originalValues = new double[testInstances.numInstances()];
//		for(int i=0;i<testInstances.numInstances();i++) {
//			Instance instance = testInstances.instance(i);
//			
//			originalValues[i] = instance.classValue();
//			predictedValues[i] = y[i];
//
//		}
//		
//		double rSquared = StatUtils.rSquared(originalValues, predictedValues);
//		System.out.println("R-squared (test): "+rSquared);
		
		
		
	}
	
	public static void main(String[] args) throws Exception {
		NumberOfResultsPredictorKNN predictor = new NumberOfResultsPredictorKNN();
		predictor.loadData();
		predictor.trainKNNRegression();
		predictor.crossValidation();
		predictor.testModel();
	}
}

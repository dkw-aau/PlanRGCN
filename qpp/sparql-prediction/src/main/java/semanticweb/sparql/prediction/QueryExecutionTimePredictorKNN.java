package semanticweb.sparql.prediction;

import java.io.IOException;
import java.util.HashMap;
import java.util.concurrent.TimeUnit;

import com.google.common.base.Stopwatch;

//import org.apache.jena.ext.com.google.common.base.Stopwatch;
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

public class QueryExecutionTimePredictorKNN {
	//public static String CONFIG_FILE = System.getProperty("user.home")+"/Documents/code/query-performance/config-knn-6000.prop";
	//public static String CONFIG_FILE = System.getProperty("user.home")+"/Documents/code/query-performance/config-knn-dbpsb-test.prop";
	
	public static String CONFIG_FILE = System.getProperty("user.home")+"/Documents/code/query-performance/config-dbpsb-k10.prop";
	//public static String CONFIG_FILE = System.getProperty("user.home")+"/Documents/code/query-performance/config-dbpsb-k10.prop";
	//public static String CONFIG_FILE = System.getProperty("user.home")+"/Documents/code/query-performance/config-dbpsb-k15.prop";
	//public static String CONFIG_FILE = System.getProperty("user.home")+"/Documents/code/query-performance/config-dbpsb-k20.prop";
	//public static String CONFIG_FILE = System.getProperty("user.home")+"/Documents/code/query-performance/config-dbpsb-k25.prop";
	
	
	ProjectConfiguration config;
	Instances trainingInstances;
	Instances validationInstances;
	Instances testInstances;
	AttributeFilterMeta trainingFeaturesMeta;
	IBk knnModel;
	int K4KNN = 5;
	
	public QueryExecutionTimePredictorKNN() throws IOException {
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
		Instances trainingExecutionTimeInstances = WekaUtils.loadCSV(config.getTrainingQueryExecutionTimesFile());
		
		trainingFeaturesMeta = WekaUtils.refineInstances(trainingAlgebraFeatureInstances, trainingSimVecFeatureInstances);
		
		Instances features = trainingFeaturesMeta.getInstances();
		trainingInstances = WekaUtils.addNumericLablesForRegression(features, trainingFeaturesMeta, trainingExecutionTimeInstances); 
		//regressionTrainingDatasets = new HashMap<Integer, Instances>();
	}	
	
	public void loadValidationtData() throws Exception {
		Instances validationAlgebraFeatureInstances = WekaUtils.loadCSV(config.getValidationAlgebraFeaturesFile());
		Instances validationSimVecFeatureInstances = WekaUtils.loadCSV(config.getValidationSimilarityVectorFeatureFile());
		Instances validationtExecutionTimeInstances = WekaUtils.loadCSV(config.getValidationQueryExecutionTimesFile());
		
		Instances features = WekaUtils.refineInstances(validationAlgebraFeatureInstances, validationSimVecFeatureInstances,trainingFeaturesMeta);
		//WekaUtils.refineInstances(algebraInstances, simVecInstances, meta)
		
		//Instances features = featuresMeta.getInstances();
		validationInstances = WekaUtils.addNumericLablesForRegression(features, trainingFeaturesMeta, validationtExecutionTimeInstances); 
		
	}
	
	public void loadTestData() throws Exception {
		Instances testAlgebraFeatureInstances = WekaUtils.loadCSV(config.getTestAlgebraFeaturesFile());
		Instances testSimVecFeatureInstances = WekaUtils.loadCSV(config.getTestSimilarityVectorFeatureFile());
		Instances testExecutionTimeInstances = WekaUtils.loadCSV(config.getTestQueryExecutionTimesFile());
		
		Instances features = WekaUtils.refineInstances(testAlgebraFeatureInstances, testSimVecFeatureInstances,trainingFeaturesMeta);
		//WekaUtils.refineInstances(algebraInstances, simVecInstances, meta)
		
		//Instances features = featuresMeta.getInstances();
		testInstances = WekaUtils.addNumericLablesForRegression(features, trainingFeaturesMeta, testExecutionTimeInstances); 
		
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
		Stopwatch watch = Stopwatch.createStarted();
		QueryExecutionTimePredictorKNN predictor = new QueryExecutionTimePredictorKNN();
		predictor.loadData();
		
		watch.start();
		predictor.trainKNNRegression();
		watch.stop();
		System.out.println("Total training time k-NN + algebra + graph pattern: "+watch.elapsed(TimeUnit.MILLISECONDS)+" ms");
		predictor.crossValidation();
		
		watch.reset();
        watch.start();
		predictor.testModel();
        watch.stop();
        System.out.println("Total prediction time for test dataset k-NN + algebra + graph pattern: "+watch.elapsed(TimeUnit.MILLISECONDS)+" ms");
		
	}
}

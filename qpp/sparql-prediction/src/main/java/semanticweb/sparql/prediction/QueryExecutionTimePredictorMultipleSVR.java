package semanticweb.sparql.prediction;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.Map;

import org.apache.commons.collections.map.DefaultedMap;

import semanticweb.sparql.config.ProjectConfiguration;
import semanticweb.sparql.utils.AttributeFilterMeta;
import semanticweb.sparql.utils.GeneralUtils;
import semanticweb.sparql.utils.StatUtils;
import semanticweb.sparql.utils.WekaUtils;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.functions.SMOreg;
import weka.classifiers.functions.supportVector.Kernel;
import weka.classifiers.functions.supportVector.RBFKernel;
import weka.classifiers.functions.supportVector.RegSMOImproved;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.Utils;

public class QueryExecutionTimePredictorMultipleSVR {

	TimeClusterer timeClusterer;
	TimeClassClassifier timeClassifier;
	ProjectConfiguration config;
	Instances trainingInstances;
	Instances testInstances;
	AttributeFilterMeta trainingFeaturesMeta;
	
	Map<Integer, Regression4TimeClass> regressions;
	Map<Integer, Instances> regressionTrainingDatasets;
	int K;
	
	
	public QueryExecutionTimePredictorMultipleSVR(String configFile) throws Exception{
		
		config = new ProjectConfiguration(configFile);
		timeClusterer = new TimeClusterer(config);
		timeClassifier = new TimeClassClassifier(config);
		regressions = new HashMap<Integer, QueryExecutionTimePredictorMultipleSVR.Regression4TimeClass>();
		

	}
	
	public void loadTrainingData() throws Exception {
		Instances trainingAlgebraFeatureInstances = WekaUtils.loadCSV(config.getTrainingAlgebraFeaturesFile());
		Instances trainingSimVecFeatureInstances = WekaUtils.loadCSV(config.getTrainingSimilarityVectorfeatureFile());
		Instances trainingExecutionTimeInstances = WekaUtils.loadCSV(config.getTrainingQueryExecutionTimesFile());
		
		trainingFeaturesMeta = WekaUtils.refineInstances(trainingAlgebraFeatureInstances, trainingSimVecFeatureInstances);
		
		Instances features = trainingFeaturesMeta.getInstances();
		trainingInstances = WekaUtils.addNumericLablesForRegression(features, trainingFeaturesMeta, trainingExecutionTimeInstances); 
		regressionTrainingDatasets = new HashMap<Integer, Instances>();
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
	
	class Regression4TimeClass {
		Instances training;
		SMOreg svr;
		int classId;
		Instances tInstances;
		boolean trained = false;
		public Regression4TimeClass(Instances tInstances, double C, double G) throws Exception {
			this.tInstances = tInstances;
			svr = new SMOreg();
			svr.setC(C);
			
			Kernel kernel = new RBFKernel(tInstances,250007,G);
			svr.setKernel(kernel);
			
			RegSMOImproved regOptimizer = new RegSMOImproved();
			regOptimizer.setOptions(Utils.splitOptions("-L 0.0010 -W 1 -P 1.0E-12 -T 0.0010"));
			
			
			SelectedTag tag = new SelectedTag(SMOreg.FILTER_NONE, SMOreg.TAGS_FILTER);
			svr.setFilterType(tag);
			
			
			svr.setRegOptimizer(regOptimizer);
			
		}
		public void train() throws Exception {
			svr.buildClassifier(tInstances);
			trained = true;
		}
		public boolean isTrained() {
			return trained;
		}
		
		public void printStats() throws Exception {
			Evaluation eval = new Evaluation(tInstances);
			eval.evaluateModel(svr, tInstances);
			
			System.out.println(eval.toSummaryString("\nTraining Results\n======\n", false));			
		}
		
		public double predictExecutionTime(Instance instance) throws Exception {
			return svr.classifyInstance(instance);
		}
		
	}
	class RegressionParams {
		double[] cValues;
		double[] gValues;
		public RegressionParams(int K) {
			cValues = new double[K];
			gValues = new double[K];
			//set defaults
			for(int i=0;i<K;i++) {
				cValues[i] = 300.0;
				gValues[i] = 0.3;
			}
		}
		public void setC(int classId,double C) {
			cValues[classId] = C;
		}
		public void setG(int classId,double G) {
			gValues[classId] = G;
		}
		
		public double getC(int classId) {
			return cValues[classId];
		}
		
		public double getG(int classId) {
			return gValues[classId];
		}

	
	}
	
	public void configureRegressions() throws Exception {
		K = timeClusterer.getNumberOfClusters();
		

		
		
		int[] trainingAssignments = timeClusterer.getClusterAssignments();
		
		
		for(int i=0;i<trainingAssignments.length;i++) {
			int k = trainingAssignments[i];
			Instance currentInstance = trainingInstances.instance(i);
			Instances instancesOfk = null;
			if(regressionTrainingDatasets.containsKey(k))
				instancesOfk = regressionTrainingDatasets.get(k);
			else {
				instancesOfk = new Instances(trainingInstances,0,0);
				//instancesOfk = new Instances(""+k, attInfo, 0);
				regressionTrainingDatasets.put(k, instancesOfk);
			}
			instancesOfk.add(currentInstance);
		}
		
		
		RegressionParams params = new RegressionParams(K);
		
		for(int k=0;k<K;k++) {
			
			int classId = k;
			
			Instances instancesOfk = regressionTrainingDatasets.get(classId);
			
			//System.out.println("Total instances of class "+classId+" :"+instancesOfk.numInstances());
			Regression4TimeClass reg = new Regression4TimeClass(instancesOfk,params.getC(classId), params.getG(classId));
			
			
			regressions.put(classId, reg);
			
			/*if(instancesOfk.numInstances()==1) {
				//Instance dummy = (Instance) instancesOfk.firstInstance().copy();
				//instancesOfk.add(dummy); //doesn't work because All class values are the same. At least two class values should be different
				continue;
			}
			reg.train();
			reg.printStats();*/
		
		}
	}
	
	
	
	public void trainRegressions() throws Exception{
		loadTrainingData();
		
		configureRegressions();
		
		for(int k=0;k<K;k++) {
			
			int classId = k;
			
			Instances instancesOfk = regressionTrainingDatasets.get(classId);
			
			System.out.println("Total instances of class "+classId+" :"+instancesOfk.numInstances());
			Regression4TimeClass reg = regressions.get(classId);
			
			
			
			/*//must be handled to deal with small datasets with single instances in a cluster
			if(instancesOfk.numInstances()==1) {
				//Instance dummy = (Instance) instancesOfk.firstInstance().copy();
				//instancesOfk.add(dummy); //doesn't work because All class values are the same. At least two class values should be different
				continue;
			}*/
			reg.train();
			reg.printStats();
		
		}
		
		double[] predictedValues = new double[trainingInstances.numInstances()];
		double[] originalValues = new double[trainingInstances.numInstances()];
		
		//int[] predictedClasses = new int[trainingInstances.numInstances()];
		for(int i=0;i<trainingInstances.numInstances();i++) {
			Instance instance = trainingInstances.instance(i);
			
			originalValues[i] = instance.classValue();
			predictedValues[i] = predictExecutionTime(instance);
			//predictedClasses[i] = timeClassifier.classifyInstance(instance); 
		}
		//System.out.println("Original:"+Arrays.toString(originalValues));
		//System.out.println("Predicted:"+Arrays.toString(predictedValues));
		
		
		//System.out.println("Original class:"+Arrays.toString(timeClusterer.getClusterAssignments()));
		//System.out.println("Predicted class:"+Arrays.toString(predictedClasses));
		GeneralUtils.saveExecutionTimePredictions(predictedValues, config.getTrainingQueryExecutionTimesPredictedFile());
		
		//double rSquared = StatUtils.rSquared(originalValues, predictedValues);
		double cCoeff = StatUtils.correlationCoefficient(originalValues, predictedValues);
		System.out.println("R-squared (training): "+cCoeff*cCoeff);	
	}
	
	public void testExecutionTimePredictions() throws Exception {
		
		loadTestData();
		double[] predictedValues = new double[testInstances.numInstances()];
		double[] originalValues = new double[testInstances.numInstances()];
		
		
		for(int i=0;i<testInstances.numInstances();i++) {
			Instance instance = testInstances.instance(i);
			
			originalValues[i] = instance.classValue();
			predictedValues[i] = predictExecutionTime(instance);

		}
		//System.out.println("Original:"+Arrays.toString(originalValues));
		//System.out.println("Predicted:"+Arrays.toString(predictedValues));
		
		
		GeneralUtils.saveExecutionTimePredictions(predictedValues, config.getTestQueryExecutionTimesPredictedFile());
		
		//double rSquared = StatUtils.rSquared(originalValues, predictedValues);
		double cCoeff = StatUtils.correlationCoefficient(originalValues, predictedValues);
		System.out.println("R-squared (test): "+cCoeff*cCoeff);		
	}
	

	
	public double predictExecutionTime(Instance instance) throws Exception {
		
		//Instance inst = new DenseInstance(3); 
		//Instance nInstance = new Instance(instance);
		
		//String class_attName
		int classIndex = instance.numAttributes()-1;
		//Instance newInstance = new Instance(instance);
		Instance newInstance = new DenseInstance(instance);
		
		//newInstance.setValue(classIndex, "1");
		
		ArrayList<Attribute> attInfo = timeClassifier.getDatasetAtrributes();
		Instances dataUnlabeled = new Instances("single", attInfo, 0);
		dataUnlabeled.add(newInstance);
		dataUnlabeled.setClassIndex(dataUnlabeled.numAttributes()-1);
		dataUnlabeled.firstInstance().setValue(classIndex, "1");
		int classId = timeClassifier.classifyInstance(dataUnlabeled.firstInstance());
		
		
		double predictedTime = regressions.get(classId).predictExecutionTime(instance);
		return predictedTime;
	}
	
	
	
	public void trainClassifiers() throws Exception {
		timeClusterer.processTrainingDataXmeans();
		timeClusterer.processValidationDataXMeans();
		timeClusterer.processTestDataXmeans();
		
		
		timeClassifier.trainClassifier(config.getTrainingTimeClassXmeansFile());
		timeClassifier.processValidationDataset();
		timeClassifier.processTestDataset();
	}
	public static void test(String configFile) throws Exception {
		QueryExecutionTimePredictorMultipleSVR predictor = new QueryExecutionTimePredictorMultipleSVR(configFile);


		predictor.trainClassifiers();


		predictor.trainRegressions();
		predictor.testExecutionTimePredictions();

	}

	

	
	public static void main(String[] args) throws Exception {
		String configFile = System.getenv("CONFIG_FILE");
		test(configFile);
		//predictor.loadTrainingData();
		//predictor.loadTestData();
	}

}

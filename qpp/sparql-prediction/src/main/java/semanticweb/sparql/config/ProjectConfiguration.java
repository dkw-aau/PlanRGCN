package semanticweb.sparql.config;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.Properties;

public class ProjectConfiguration {
	//public static String CONFIG_FILE = System.getProperty("user.home")+"/Documents/code/query-performance/config-6000.prop";
	//public static String CONFIG_FILE = System.getProperty("user.home")+"/Documents/code/query-performance/config-knn-dbpsb-test.prop";
	
	//public static String CONFIG_FILE = System.getProperty("user.home")+"/Documents/code/query-performance/config-dbpsb-k5.prop";
	//public static String CONFIG_FILE = System.getProperty("user.home")+"/Documents/code/query-performance/config-dbpsb-k10.prop";
	//public static String CONFIG_FILE = System.getProperty("user.home")+"/Documents/code/query-performance/config-dbpsb-k15.prop";
	////public static String CONFIG_FILE = System.getProperty("user.home")+"/Documents/GitHub/query-performance/config-dbpsb-k20.prop";
	//public static String CONFIG_FILE = "/Users/abirammohanaraj/Documents/GitHub/query-performance/config-dbpsb-k20.prop";
	public static String CONFIG_FILE = "/Users/abirammohanaraj/Documents/GitHub/query-performance/config_lsq_20.prop" ;
	static{CONFIG_FILE = System.getenv("CONFIG_FILE");}
	//public static String CONFIG_FILE = System.getProperty("user.home")+"/Documents/code/query-performance/config-dbpsb-k25.prop";
	
	private String configFile;
	private String distanceMatrixFile = null;
	private String trainingQueryExecutionTimesFile = null;
	private String validationQueryExecutionTimesFile;
	private String testQueryExecutionTimesFile;
	
	private String trainingQueryFile = null;
	private String validationQueryFile = null;
	private String testQueryFile = null;
	
	
	
	private String trainingARFFFile;
	private String validationARFFFile;
	private String testARFFFile;	
	
	private int numberOfClusters = 2;
	private int xMeansMaxK = 2;
	private Properties prop;
	//private String trainingSimilarityVectorfeatureKmeansFile;
	//private String validationSimilarityVectorFeatureKmeansFile;
	//private String testSimilarityVectorFeatureKmeansFile;
	
	private String trainingClassFeatureKmeansFile;
	private String validationClassFeatureKmeansFile;
	private String testClassVectorFeatureKmeansFile;
	
	private String trainingTimeClassKmeansFile;
	private String validationTimeClassKmeansFile;
	private String testTimeClassKmeansFile;

	
	private String trainingClassFeatureXmeansFile;
	private String validationClassFeatureXmeansFile;
	private String testClassVectorFeatureXmeansFile;
	
	private String trainingTimeClassXmeansFile;
	private String validationTimeClassXmeansFile;
	private String testTimeClassXmeansFile;	
	
	
	private String trainingAlgebraFeaturesFile;
	private String validationAlgebraFeaturesFile;
	private String testAlgebraFeaturesFile;
	
	private String trainingSimilarityVectorfeatureFile;
	private String validationSimilarityVectorFeatureFile;
	private String testSimilarityVectorFeatureFile;

	
	private String trainingQueryExecutionTimesPredictedFile;
	private String validationQueryExecutionTimesPredictedFile;
	private String testQueryExecutionTimesPredictedFile;
	
	
	private String  trainingNumberOfRecordsFile;
	private String  validationNumberOfRecordsFile;
	private String testNumberOfRecordsFile;
	
	public ProjectConfiguration() throws IOException{
		this(ProjectConfiguration.CONFIG_FILE);

	}
	
	public ProjectConfiguration(String cFile) throws IOException {
		this.configFile = cFile;
		prop = new Properties();
		prop.load(new FileInputStream(configFile));
		loadConfig();
	}
	
	public void loadConfig() {
		
		distanceMatrixFile = prop.getProperty("TrainingDistanceHungarianMatrix");
	
		numberOfClusters = Integer.valueOf(prop.getProperty("K"));
		
		
		trainingQueryFile = prop.getProperty("TrainingQuery");
		validationQueryFile = prop.getProperty("ValidationQuery");
		testQueryFile = prop.getProperty("TestQuery");
		
		trainingSimilarityVectorfeatureFile = prop.getProperty("TrainingSimilarityVectorfeature");
		validationSimilarityVectorFeatureFile = prop.getProperty("ValidationSimilarityVectorFeature");
		testSimilarityVectorFeatureFile = prop.getProperty("TestSimilarityVectorFeature");
		
		trainingClassFeatureKmeansFile = prop.getProperty("TrainingClassVectorfeatureKmeans");
		validationClassFeatureKmeansFile = prop.getProperty("ValidationClassVectorFeatureKmeans");
		testClassVectorFeatureKmeansFile = prop.getProperty("TestClassVectorFeatureKmeans");
		
		trainingQueryExecutionTimesFile = prop.getProperty("TrainingQueryExecutionTimes");
		
		validationQueryExecutionTimesFile = prop.getProperty("ValidationQueryExecutionTimes");
		testQueryExecutionTimesFile = prop.getProperty("TestQueryExecutionTimes");

		
		trainingTimeClassKmeansFile = prop.getProperty("TrainingTimeClassKmeans");
		validationTimeClassKmeansFile = prop.getProperty("ValidationTimeClassKmeans");
		testTimeClassKmeansFile = prop.getProperty("TestTimeClassKmeans");
		
		xMeansMaxK = Integer.valueOf(prop.getProperty("XMEANS_MAX_K"));
		
		
		trainingClassFeatureXmeansFile = prop.getProperty("TrainingClassVectorfeatureXmeans");
		validationClassFeatureXmeansFile = prop.getProperty("ValidationClassVectorFeatureXmeans");
		testClassVectorFeatureXmeansFile = prop.getProperty("TestClassVectorFeatureXmeans");
	

		trainingTimeClassXmeansFile = prop.getProperty("TrainingTimeClassXmeans");
		validationTimeClassXmeansFile = prop.getProperty("ValidationTimeClassXmeans");
		testTimeClassXmeansFile = prop.getProperty("TestTimeClassXmeans");
		
		trainingAlgebraFeaturesFile = prop.getProperty("TrainingAlgebraFeatures");
		validationAlgebraFeaturesFile = prop.getProperty("ValidationAlgebraFeatures");
		testAlgebraFeaturesFile = prop.getProperty("TestAlgebraFeatures");


		trainingQueryExecutionTimesPredictedFile = prop.getProperty("TrainingQueryExecutionTimesPredicted");
		validationQueryExecutionTimesPredictedFile = prop.getProperty("ValidationQueryExecutionTimesPredicted");
		testQueryExecutionTimesPredictedFile = prop.getProperty("TestQueryExecutionTimesPredicted");
	
		
		trainingARFFFile = prop.getProperty("TrainingARFFFile");
		validationARFFFile = prop.getProperty("ValidationARFFFile");
		testARFFFile = prop.getProperty("TestARFFFile");
		
		
		trainingNumberOfRecordsFile = prop.getProperty("TrainingNumberOfRecords");
		validationNumberOfRecordsFile = prop.getProperty("ValidationNumberOfRecords");
		testNumberOfRecordsFile = prop.getProperty("TestNumberOfRecords");		
	}	
	
	public String getTrainingNumberOfRecordsFile() {
		return trainingNumberOfRecordsFile;
	}
	public String getValidationNumberOfRecordsFile() {
		return validationNumberOfRecordsFile;
	}
	public String getTestNumberOfRecordsFile() {
		return testNumberOfRecordsFile;
	}
	public String getValidationSimilarityVectorFeatureFile() {
		return validationSimilarityVectorFeatureFile;
	}
	public String getTrainingSimilarityVectorfeatureFile() {
		return trainingSimilarityVectorfeatureFile;
	}
	public String getTestSimilarityVectorFeatureFile() {
		return testSimilarityVectorFeatureFile;
	}
	
	public String getValidationAlgebraFeaturesFile() {
		return validationAlgebraFeaturesFile;
	}
	public String getTrainingAlgebraFeaturesFile() {
		return trainingAlgebraFeaturesFile;
	}
	public String getTestAlgebraFeaturesFile() {
		return testAlgebraFeaturesFile;
	}
	
	public String getTrainingQueryExecutionTimesFile() {
		return trainingQueryExecutionTimesFile;
	}
	public String getDistanceMatrixFile() {
		return distanceMatrixFile;
	}
	
	public int getNumberOfClusters() {
		return numberOfClusters;
	}
	
	public String getTrainingQueryFile() {
		return trainingQueryFile;
	}
	/*
	public String getTestSimilarityVectorFeatureKmeansFile() {
		return testSimilarityVectorFeatureKmeansFile;
	}
	public String getTrainingSimilarityVectorfeatureKmeansFile() {
		return trainingSimilarityVectorfeatureKmeansFile;
	}
	
	public String getValidationSimilarityVectorFeatureKmeansFile() {
		return validationSimilarityVectorFeatureKmeansFile;
	}
	*/
	public String getValidationQueryFile() {
		return validationQueryFile;
	}
	public String getTestQueryFile() {
		return testQueryFile;
	}
	
	public String getTestClassVectorFeatureKmeansFile() {
		return testClassVectorFeatureKmeansFile;
	}
	public String getTrainingClassFeatureKmeansFile() {
		return trainingClassFeatureKmeansFile;
	}
	public String getValidationClassFeatureKmeansFile() {
		return validationClassFeatureKmeansFile;
	}
	
	public String getTestQueryExecutionTimesFile() {
		return testQueryExecutionTimesFile;
	}
	public String getValidationQueryExecutionTimesFile() {
		return validationQueryExecutionTimesFile;
	}
	
	
	public String getTrainingTimeClassKmeansFile() {
		return trainingTimeClassKmeansFile;
	}
	public String getValidationTimeClassKmeansFile() {
		return validationTimeClassKmeansFile;
	}
	public String getTestTimeClassKmeansFile() {
		return testTimeClassKmeansFile;
	}
	
	public String getValidationQueryExecutionTimesPredictedFile() {
		return validationQueryExecutionTimesPredictedFile;
	}
	public String getTrainingQueryExecutionTimesPredictedFile() {
		return trainingQueryExecutionTimesPredictedFile;
	}
	public String getTestQueryExecutionTimesPredictedFile() {
		return testQueryExecutionTimesPredictedFile;
	}
	
	public String getValidationARFFFile() {
		return validationARFFFile;
	}
	public String getTrainingARFFFile() {
		return trainingARFFFile;
	}
	public String getTestARFFFile() {
		return testARFFFile;
	}
	
	public int getxMeansMaxK() {
		return xMeansMaxK;
	}
	
	public String getValidationTimeClassXmeansFile() {
		return validationTimeClassXmeansFile;
	}
	public String getValidationClassFeatureXmeansFile() {
		return validationClassFeatureXmeansFile;
	}
	public String getTrainingTimeClassXmeansFile() {
		return trainingTimeClassXmeansFile;
	}
	public String getTrainingClassFeatureXmeansFile() {
		return trainingClassFeatureXmeansFile;
	}
	public String getTestTimeClassXmeansFile() {
		return testTimeClassXmeansFile;
	}
	public String getTestClassVectorFeatureXmeansFile() {
		return testClassVectorFeatureXmeansFile;
	}
	
	public static String getAlgebraFeatureHeader() {
		return "triple,bgp,join,leftjoin,union,filter,graph,extend,minus,path*,pathN*,path+,pathN+,notoneof,tolist,order,project,distinct,reduced,multi,top,group,assign,sequence,slice,treesize";
	}
	
	public static String getPatternClusterSimVecFeatureHeader(int dim) {
		String out = "";
		for(int i=0;i<dim;i++) {
			if(i!=0) {
				out += ",";
			}
			out += ("pcs"+i);
		}
		return out;
	}

	public static String getTimeClusterBinaryVecFeatureHeader(int dim) {
		String out = "";
		for(int i=0;i<dim;i++) {
			if(i!=0) {
				out += ",";
			}
			out += ("tcb"+i);
		}
		return out;
	}
	
	
	public static String getTimClusterLabelHeader() {
		return "time_class";
	}
	
	public static String getExecutionTimeHeader() {
		return "ex_time";
	}
}

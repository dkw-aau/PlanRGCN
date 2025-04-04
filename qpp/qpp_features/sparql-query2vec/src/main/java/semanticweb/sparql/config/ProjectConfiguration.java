package semanticweb.sparql.config;

import java.io.*;
import java.util.ArrayList;
import java.util.Properties;

public class ProjectConfiguration {
	public static String CONFIG_FILE = "/home/daniel/Documentos/ML/rhassan/graph-edit-distance/data/config/config-100.prop";

	private Properties prop;

	public ProjectConfiguration() throws IOException{
		this(ProjectConfiguration.CONFIG_FILE);

	}
	
	public ProjectConfiguration(String cFile) throws IOException {
		prop = new Properties();
		prop.load(new FileInputStream(cFile));
		loadConfig();
	}
	
	private void loadConfig() {

		String distanceMatrixFile = prop.getProperty("TrainingDistanceHungarianMatrix");

		int numberOfClusters = Integer.parseInt(prop.getProperty("K"));

		String trainingQueryFile = prop.getProperty("TrainingQuery");

		String trainingSimilarityVectorfeatureFile = prop.getProperty("TrainingSimilarityVectorfeature");

		String trainingClassFeatureKmeansFile = prop.getProperty("TrainingClassVectorfeatureKmeans");

		String trainingQueryExecutionTimesFile = prop.getProperty("TrainingQueryExecutionTimes");

		String trainingTimeClassKmeansFile = prop.getProperty("TrainingTimeClassKmeans");

		String trainingClassFeatureXmeansFile = prop.getProperty("TrainingClassVectorfeatureXmeans");

		String trainingTimeClassXmeansFile = prop.getProperty("TrainingTimeClassXmeans");

		String trainingAlgebraFeaturesFile = prop.getProperty("TrainingAlgebraFeatures");

		String trainingQueryExecutionTimesPredictedFile = prop.getProperty("TrainingQueryExecutionTimesPredicted");

		String trainingARFFFile = prop.getProperty("TrainingARFFFile");

		String trainingNumberOfRecordsFile = prop.getProperty("TrainingNumberOfRecords");
	}

	/**
	 * Get a features list specified in a file as parameter.
	 * @param url {@link String} Path of features file to parse.
	 * @return String[]
	 */
	public static String[] getAlgebraFeatureHeader(String url) {
		String row;
		ArrayList<String> result = new ArrayList<>();
		try {
			BufferedReader reader = new BufferedReader(new FileReader(url));
			while ((row = reader.readLine()) != null) {
				if(row.equals( ""))
					continue;
				result.add(row);
			}
			reader.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
			System.out.println("The features file was not found");
		}
		catch (IOException e) {
			e.printStackTrace();
		}

		return result.toArray(new String[0]);
	}

	/**
	 * Get a list of fixed features instead as the list specified in a file.
	 * @return String[]
	 */
	public static Object[] getAlgebraFeatureHeader() {
		String feats = "triple,bgp,join,leftjoin,union,filter,graph,extend,minus,path*,pathN*,path+,pathN+,notoneof,tolist,order,project,distinct,reduced,multi,top,group,assign,sequence,slice,treesize";
		return feats.split(",");
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

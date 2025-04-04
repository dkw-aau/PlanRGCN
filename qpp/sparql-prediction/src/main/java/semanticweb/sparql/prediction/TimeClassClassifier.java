package semanticweb.sparql.prediction;

import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import semanticweb.sparql.config.ProjectConfiguration;

import com.google.common.primitives.Ints;

import weka.classifiers.Evaluation;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.trees.m5.CorrelationSplitInfo;
import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.AddValues;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.RemoveUseless;
import weka.filters.unsupervised.attribute.Standardize;

public class TimeClassClassifier {


	Instances trainingInstances;
	Instances validationInstances;
	Instances testInstances;
	ProjectConfiguration config;
	LibSVM svm;
	Map<String,Integer> attributeIndex;
	Standardize stdFilter;
	ArrayList<Attribute> datasetAttributes = new ArrayList<Attribute>();
	
	
	
	int[] removedAttributes;
	public TimeClassClassifier() throws Exception {
		
		svm = new LibSVM();
		config = new ProjectConfiguration();
		attributeIndex = new HashMap<String, Integer>();
		

		
		
		
		
		//trainClassifier(labelFilePath);
	}
	public TimeClassClassifier(ProjectConfiguration i) throws Exception {

		svm = new LibSVM();
		config = i;
		attributeIndex = new HashMap<String, Integer>();






		//trainClassifier(labelFilePath);
	}
	/*
	public Instances loadAndrefineInstances(String algebraFeatureFile, String simVecFeatureFile) throws Exception{
		return loadAndrefineInstances(algebraFeatureFile, simVecFeatureFile,"",false);
	}*/
	
	public Instances loadAndrefineInstances(String algebraFeatureFile, String simVecFeatureFile,String labelFilePath) throws Exception{
		return loadAndrefineInstances(algebraFeatureFile, simVecFeatureFile,labelFilePath,false);
	}	
	
	public Instances loadAndrefineInstances(String algebraFeatureFile, String simVecFeatureFile, String labelFilePath, boolean trainingDataSet) throws Exception{
		CSVLoader algebraFeaturesCSVLoader = new CSVLoader();
		System.out.println(algebraFeatureFile);
		algebraFeaturesCSVLoader.setFile(new File(algebraFeatureFile));
		Instances algebraInstances = algebraFeaturesCSVLoader.getDataSet(); 
		
		CSVLoader simVecFeaturesCSVLoader = new CSVLoader();
		simVecFeaturesCSVLoader.setFile(new File(simVecFeatureFile));
		Instances simVecInstances = simVecFeaturesCSVLoader.getDataSet();
		


		
		//merge algebra and simvec
		Instances mergedAlgSimVec = Instances.mergeInstances(algebraInstances, simVecInstances);
		

		
		if(trainingDataSet) {
			//List<Attribute> atts = new ArrayList<Attribute>();
			//mergedAlgSimVecClass.attribute(0).
			for(int i=0;i < mergedAlgSimVec.numAttributes();i++) {
				attributeIndex.put(mergedAlgSimVec.attribute(i).name(), i);
			}
		}
		
		Instances mergedUselessFilteredAlgSimVecClass = null;
		if(trainingDataSet) {
			//filter out useless
			RemoveUseless removeUselessFilter = new RemoveUseless();
			removeUselessFilter.setMaximumVariancePercentageAllowed(99.0);
			removeUselessFilter.setInputFormat(mergedAlgSimVec);
			mergedUselessFilteredAlgSimVecClass = Filter.useFilter(mergedAlgSimVec, removeUselessFilter);
		} else {
			//mergedUselessFilteredAlgSimVecClass = mergedAlgSimVec.
			
			Remove removeFilter = new Remove();
			removeFilter.setAttributeIndicesArray(removedAttributes);
			removeFilter.setInputFormat(mergedAlgSimVec);
			mergedUselessFilteredAlgSimVecClass = Filter.useFilter(mergedAlgSimVec, removeFilter);
		}
		
		

		
		//initialize only with training set to reuse mu and sigma for other datasets
		if(trainingDataSet) {
			//standardize
			stdFilter = new Standardize();
		
			//stdFilter.setIgnoreClass(true);
			
		}
		stdFilter.setInputFormat(mergedUselessFilteredAlgSimVecClass);
		
		Instances stdFilterdInstances = Filter.useFilter(mergedUselessFilteredAlgSimVecClass, stdFilter); 
		
		

		

		
		Instances filteredlabelInstances = null;
		Instances finalCleaned = null;
		
		if(trainingDataSet || labelFilePath.isEmpty()==false) {
			System.out.println("loading class labes from:" +labelFilePath);
			CSVLoader labelCSVLoader = new CSVLoader();
			labelCSVLoader.setFile(new File(labelFilePath));
			Instances labelInstances = labelCSVLoader.getDataSet();
			
			//convert class labes to nominal
			NumericToNominal classNominalFilter = new NumericToNominal();		
			classNominalFilter.setInputFormat(labelInstances);
			classNominalFilter.setAttributeIndices("last");
			filteredlabelInstances = Filter.useFilter(labelInstances, classNominalFilter);
			

		} /*else {
			FastVector atts = new FastVector();
			Attribute at = trainingInstances.classAttribute().copy(trainingInstances.classAttribute().name());
			atts.addElement(at);
			
			filteredlabelInstances = new Instances("dummyLabels", atts, 0);
			
			for(int di = 0;di < stdFilterdInstances.numInstances();di++) {
				Instance ins = new Instance(1);
				ins.setValue(trainingInstances.classAttribute(), "0");
				filteredlabelInstances.add(ins);
			}
		}*/
		
		
		//to add all the available classes in the definition, otherwise just considers the available classes in a dataset
		if(trainingDataSet==false) {
			Attribute classAt = trainingInstances.classAttribute();
			int numOfAttValues = classAt.numValues();
			String attValues = "";
			for(int nai=0;nai<numOfAttValues;nai++) {
				if(nai!=0) {
					attValues +=",";
				}
				attValues += classAt.value(nai);
				
			}
			
			AddValues addValFilter = new AddValues();
			addValFilter.setSort(true);
			addValFilter.setAttributeIndex("last");
			addValFilter.setLabels(attValues);
			addValFilter.setInputFormat(filteredlabelInstances);
			
			filteredlabelInstances = Filter.useFilter(filteredlabelInstances, addValFilter);
		}


		//merge algebra and simvec and class labels
		finalCleaned = Instances.mergeInstances(stdFilterdInstances, filteredlabelInstances);
		int classAttIndex = finalCleaned.numAttributes()-1;
		finalCleaned.setClassIndex(classAttIndex);
		attributeIndex.put(finalCleaned.attribute(classAttIndex).name(), classAttIndex);		
		
		if(trainingDataSet) {
			
		
			//record removed attributes/columns in a matrix
			Set<String> selectedAtt = new HashSet<String>();
			
			for(int i=0;i < finalCleaned.numAttributes();i++) {
				selectedAtt.add(finalCleaned.attribute(i).name());
			}
			
			List<Integer> deletedAttIndex = new ArrayList<Integer>();
			for(Entry<String, Integer> e:attributeIndex.entrySet()) {
				if(selectedAtt.contains(e.getKey())==false) {
					deletedAttIndex.add(e.getValue());
				}
			}
			removedAttributes = Ints.toArray(deletedAttIndex);	
		}

		return finalCleaned;
		
		
	}
	
	public void loadAndCleanTrainingData(String labelFilePath) throws Exception {
		
		trainingInstances = loadAndrefineInstances(config.getTrainingAlgebraFeaturesFile(), config.getTrainingSimilarityVectorfeatureFile(),labelFilePath,true);
		
		//ArrayList<Attribute> atts = new ArrayList<Attribute>();
		
		for(int i=0;i<trainingInstances.numAttributes();i++) {
			Attribute originalAt = trainingInstances.attribute(i);
			Attribute at = originalAt.copy(originalAt.name());
			datasetAttributes.add(at);
			//System.out.println(at);
		}
		
		//int classAttIndex = trainingInstances.classIndex();
		
		

		
		//System.out.println(instances.numInstances());
		//System.out.println(instances);		
	}
	public void trainClassifier(String labelFilePath) throws Exception {
		
		loadAndCleanTrainingData(labelFilePath);
		
		double C = 45.0;
		//double C = 260.0;
		double G = 1.0;
		double N = 0.01;
		//double N = 0.4;
		
		
		svm.setCost(C);
		svm.setGamma(G);
		svm.setNu(N);
		//svm.setProbabilityEstimates(true);
		svm.buildClassifier(trainingInstances);
		
		Evaluation eval = new Evaluation(trainingInstances);
		eval.evaluateModel(svm, trainingInstances);
		System.out.println(eval.toSummaryString("\nTraining Results\n======\n", false));	
		
		System.out.println(eval.toMatrixString());
		
	}
	
	public void processValidationDataset() throws Exception {
		
		validationInstances = loadAndrefineInstances(config.getValidationAlgebraFeaturesFile(), config.getValidationSimilarityVectorFeatureFile(),config.getValidationTimeClassXmeansFile());
		
		createClassVectorFeatures(validationInstances,config.getValidationClassFeatureXmeansFile());

		//test
		//System.out.println("Predicted:"+svm.classifyInstance(validationInstances.lastInstance()));
		//System.out.println("Given:"+validationInstances.lastInstance().classValue());
		
		Evaluation eval = new Evaluation(trainingInstances); //TODO: maybe validationInstances instead.
		eval.evaluateModel(svm, validationInstances);
		
		System.out.println(eval.toSummaryString("\nValidation Results\n======\n", false));
		System.out.println(eval.toMatrixString());
	}
	
	
	public void createClassVectorFeatures(Instances instances, String outFile) throws Exception {
		PrintStream ps = new PrintStream(outFile);
		
		Enumeration<Instance> en =  instances.enumerateInstances();
		DecimalFormat df=new DecimalFormat("0.000");
		int count = 0;
		while(en.hasMoreElements()) {
			Instance ins = en.nextElement();
			double[] vec = svm.distributionForInstance(ins);
			
			if(count==0) {
				ps.println(ProjectConfiguration.getTimeClusterBinaryVecFeatureHeader(vec.length));
			}
			
			int cc = 0;
			String outString = "";
			for(double c:vec) {
				
				if(cc!=0) {
					outString += ",";
				}
				
				outString += df.format(c);
				cc++;
			}			
			
			ps.println(outString);
			count++;
		}
		
		ps.close();
		
	}
	
	public void processTestDataset() throws Exception {
		
		testInstances = loadAndrefineInstances(config.getTestAlgebraFeaturesFile(), config.getTestSimilarityVectorFeatureFile(),config.getTestTimeClassXmeansFile());
		
		createClassVectorFeatures(testInstances,config.getTestClassVectorFeatureXmeansFile());

		//test
		//System.out.println("Predicted:"+svm.classifyInstance(testInstances.lastInstance()));
		//System.out.println("Given:"+testInstances.lastInstance().classValue());
		
		Evaluation eval = new Evaluation(trainingInstances);
		eval.evaluateModel(svm, testInstances);
		
		System.out.println(eval.toSummaryString("\nTest Results\n======\n", false));
		System.out.println(eval.toMatrixString());
	}	
	
	public int classifyInstance(Instance instance) throws Exception {
		return (int) svm.classifyInstance(instance);
	}
	
	/**
	public Instance refineUnseenInstance(Instance unseen) {

		for(int atInx: removedAttributes) {
			unseen.deleteAttributeAt(atInx);
		}
		return unseen;
	}*/
	
	public ArrayList<Attribute> getDatasetAtrributes() {
		
		return datasetAttributes;
	}

	
	
	public static void main(String[] args) throws Exception {
		ProjectConfiguration c = new ProjectConfiguration();
		TimeClassClassifier tc = new TimeClassClassifier();
		
		tc.trainClassifier(c.getTrainingTimeClassXmeansFile());
		
		tc.processValidationDataset();
		tc.processTestDataset();
		
		
	}

}

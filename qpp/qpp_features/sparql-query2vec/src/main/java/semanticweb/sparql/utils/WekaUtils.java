package semanticweb.sparql.utils;

import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.*;

import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.Map.Entry;

public class WekaUtils {
	
	
	
	public static Instances removeUseless(Instances inInstances) throws Exception {
		//filter out useless
		RemoveUseless removeUselessFilter = new RemoveUseless();
		removeUselessFilter.setMaximumVariancePercentageAllowed(99.0);
		removeUselessFilter.setInputFormat(inInstances);
		return Filter.useFilter(inInstances, removeUselessFilter);		
	}
	
	public static Instances removeAttributes(Instances inInstances, int[] removedAttributes) throws Exception {
		
		Remove removeFilter = new Remove();
		removeFilter.setAttributeIndicesArray(removedAttributes);
		removeFilter.setInputFormat(inInstances);
		return Filter.useFilter(inInstances, removeFilter);		
	}
	
	public static Instances standardize(Instances inInstances,Standardize stdFilter) throws Exception {
		
		
		return Filter.useFilter(inInstances, stdFilter); 
	}
	
	
	public static Instances numericToNominalLast(Instances inInstances) throws Exception {
		NumericToNominal classNominalFilter = new NumericToNominal();		
		classNominalFilter.setInputFormat(inInstances);
		classNominalFilter.setAttributeIndices("last");
		return Filter.useFilter(inInstances, classNominalFilter);		
	}
	
	/**
	 * use for training data
	 * @param instancesWithMeta
	 * @param labelInstances
	 * @return
	 * @throws Exception
	 */
	public static Instances addNumericLabelsForClassificationToTrainingData(Instances instances,AttributeFilterMeta instancesWithMeta, Instances labelInstances) throws Exception {
		//merge algebra and simvec and class labels
		//convert class labes to nominal
		Instances filteredlabelInstances = numericToNominalLast(labelInstances);
		Instances finalCleaned = addNominalLabelsForClassificationToTrainingData(instances, instancesWithMeta, filteredlabelInstances);
		return finalCleaned;

	}
	
	/**
	 * use for training data
	 * @param instancesWithMeta
	 * @param labelInstances
	 * @return
	 * @throws Exception
	 */
	
	public static Instances addNominalLabelsForClassificationToTrainingData(Instances instances, AttributeFilterMeta instancesWithMeta, Instances labelInstances) throws Exception {

		
		
		Instances finalCleaned = Instances.mergeInstances(instances, labelInstances);
		finalCleaned.setClassIndex(finalCleaned.numAttributes()-1);
		
		
		Attribute classAt = finalCleaned.classAttribute();
		int numOfAttValues = classAt.numValues();
		String attValues = "";
		for(int nai=0;nai<numOfAttValues;nai++) {
			if(nai!=0) {
				attValues +=",";
			}
			attValues += classAt.value(nai);
		}
		instancesWithMeta.setClassAtrributeValues(attValues);
		
		instancesWithMeta.setInstances(finalCleaned);
		
		return finalCleaned;
	}
	
	/**
	 * use for validation and test data
	 * @param instancesWithMeta
	 * @param labelInstances
	 * @return
	 * @throws Exception
	 */
	public static Instances addNumericLabelsForClassification(Instances instances, AttributeFilterMeta instancesWithMeta, Instances labelInstances) throws Exception {
		Instances filteredlabelInstances = numericToNominalLast(labelInstances);
		Instances finalCleaned = addNominalLabelsForClassification(instances, instancesWithMeta, filteredlabelInstances);
		return finalCleaned;
	}
	
	/**
	 * use for validation and test data
	 * @param instancesWithMeta
	 * @param labelInstances
	 * @return
	 * @throws Exception
	 */
	public static Instances addNominalLabelsForClassification(Instances instances, AttributeFilterMeta instancesWithMeta, Instances labelInstances) throws Exception {
		//to add all the available classes in the definition, otherwise just considers the available classes in a dataset
		
		AddValues addValFilter = new AddValues();
		addValFilter.setSort(true);
		addValFilter.setAttributeIndex("last");
		addValFilter.setLabels(instancesWithMeta.getClassAtrributeValues());
		addValFilter.setInputFormat(labelInstances);
		
		Instances filteredlabelInstances = Filter.useFilter(labelInstances, addValFilter);
		
		Instances finalInstances = Instances.mergeInstances(instances, filteredlabelInstances);
		finalInstances.setClassIndex(finalInstances.numAttributes()-1);
		instancesWithMeta.setInstances(finalInstances);
		return finalInstances;
	}
	
	/**
	 * use for regression training, validation, test data
	 * @param instancesWithMeta
	 * @param labelInstances
	 * @return
	 * @throws Exception
	 */
	public static Instances addNumericLablesForRegression(Instances instances, AttributeFilterMeta instancesWithMeta, Instances labelInstances) throws Exception {
		
		
		Instances finalCleaned = Instances.mergeInstances(instances, labelInstances);
		
		finalCleaned.setClassIndex(finalCleaned.numAttributes()-1);
		instancesWithMeta.setInstances(finalCleaned);
		
		
		return finalCleaned;
	}
	
	public static Instances loadCSV(String fileName) throws IOException {
		CSVLoader algebraFeaturesCSVLoader = new CSVLoader();
		algebraFeaturesCSVLoader.setFile(new File(fileName));
		return algebraFeaturesCSVLoader.getDataSet(); 
		
	}
	
	/**
	 * refines an instance and makes it like the training instances with the class attribute and a dummy value for it
	 * @param ins
	 * @param meta
	 * @return
	 * @throws Exception 
	 */
	/*//later when need to classify instance by instance for apps
	public static Instance refineUnseenInstance(Instance ins, AttributeFilterMeta meta, boolean forRegression) throws Exception {
		
		Instance in = new Instance(ins);
		int last = in.numAttributes();
		in.insertAttributeAt(last);
		if(forRegression)
			in.setValue(last, 0.0);
		else
			in.setValue(last, "0");
		
		for(int attIndx:meta.getRemovedAttributes()) {
			in.deleteAttributeAt(attIndx);
		}
		FastVector attInfo = new FastVector();
		
		for(int attIndx=0;attIndx<in.numAttributes();attIndx++) {
			attInfo.addElement(in.attribute(attIndx));
		}
		
		Instances intncs = new Instances("dummy", attInfo, 0);
		//intncs.insertAttributeAt(att, position)
		intncs.add(in);
		intncs.setClassIndex(intncs.numAttributes()-1);
		intncs = standardize(intncs, meta.getStandardizeFilter());
		
		
		return intncs.firstInstance();
	}*/
	
	/**
	 * use for validation and test data
	 * @param algebraInstances
	 * @param simVecInstances
	 * @param meta
	 * @return
	 * @throws Exception
	 */
	public static Instances refineInstances(Instances algebraInstances, Instances simVecInstances,AttributeFilterMeta meta) throws Exception {
		
				
		//merge algebra and simvec
		Instances mergedAlgSimVec = Instances.mergeInstances(algebraInstances, simVecInstances);
		

		//remove the same attributes that were removed in training data
		//System.out.println(meta);
		//System.out.println(Arrays.toString(meta.getRemovedAttributes()));
		Instances filteredAlgSimVecClass = removeAttributes(mergedAlgSimVec, meta.getRemovedAttributes());		
		

		
		//stdFilter.setInputFormat(mergedUselessFilteredAlgSimVecClass);
		
		Instances stdFilterdInstances = Filter.useFilter(filteredAlgSimVecClass, meta.getStandardizeFilter()); 
		
		return stdFilterdInstances;

	}
	
	public static Instances refineInstances(Instances algebraInstances, AttributeFilterMeta meta) throws Exception {
		
		
		//merge algebra and simvec
		//Instances mergedAlgSimVec = Instances.mergeInstances(algebraInstances, simVecInstances);
		

		//remove the same attributes that were removed in training data
		//System.out.println(meta);
		//System.out.println(Arrays.toString(meta.getRemovedAttributes()));
		Instances filteredAlgSimVecClass = removeAttributes(algebraInstances, meta.getRemovedAttributes());		
		

		
		//stdFilter.setInputFormat(mergedUselessFilteredAlgSimVecClass);
		
		Instances stdFilterdInstances = Filter.useFilter(filteredAlgSimVecClass, meta.getStandardizeFilter()); 
		
		return stdFilterdInstances;

	}	
	
	/**
	 * loading training data
	 * @param mergedAlgSimVec algebra instances normally for applying the remove useless and standardization filtering
	 * @return
	 * @throws Exception
	 */
	
	public static AttributeFilterMeta refineInstances(Instances mergedAlgSimVec) throws Exception {
		AttributeFilterMeta res = new AttributeFilterMeta();
		Map<String,Integer> attributeIndex = new HashMap<String, Integer>(); 

		
		//List<Attribute> atts = new ArrayList<Attribute>();
		//mergedAlgSimVecClass.attribute(0).
		for(int i=0;i < mergedAlgSimVec.numAttributes();i++) {
			attributeIndex.put(mergedAlgSimVec.attribute(i).name(), i);
		}
	
		
		Instances mergedUselessFilteredAlgSimVecClass = removeUseless(mergedAlgSimVec);

		

		Standardize stdFilter = new Standardize(); //kepp in meta
		stdFilter.setInputFormat(mergedUselessFilteredAlgSimVecClass); // initializing the filter once with training set
		
		Instances stdFilterdInstances = standardize(mergedUselessFilteredAlgSimVecClass,stdFilter);
		

			
		
		//record removed attributes/columns in a matrix
		Set<String> selectedAtt = new HashSet<String>();
		
		for(int i=0;i < stdFilterdInstances.numAttributes();i++) {
			selectedAtt.add(stdFilterdInstances.attribute(i).name());
		}
		
		List<Integer> deletedAttIndex = new ArrayList<>();
		for(Entry<String, Integer> e:attributeIndex.entrySet()) {
			if(selectedAtt.contains(e.getKey())==false) {
				deletedAttIndex.add(e.getValue());
			}
		}
		int[] removedAttributes = deletedAttIndex.stream().mapToInt(i -> i).toArray();
		
		res.setInstances(stdFilterdInstances);
		res.setRemovedAttributes(removedAttributes);
		res.setStandardizeFilter(stdFilter);

		return res;
		
	}
	
	/**
	 * for loading training data
	 * @param algebraInstances
	 * @param simVecInstances
	 * @return
	 * @throws Exception
	 */
	public static AttributeFilterMeta refineInstances(Instances algebraInstances, Instances simVecInstances) throws Exception{
		
		
		
		//merge algebra and simvec
		Instances mergedAlgSimVec = Instances.mergeInstances(algebraInstances, simVecInstances);
		
		return refineInstances(mergedAlgSimVec);
	}
	
	public static void saveInstances(Instances dataSet, String fileName) throws Exception{
		ArffSaver saver = new ArffSaver();
		saver.setInstances(dataSet);
		 saver.setFile(new File(fileName));
		 //saver.setDestination(new File("./data/test.arff"));   // **not** necessary in 3.5.4 and later
		 saver.writeBatch();		
	}

}

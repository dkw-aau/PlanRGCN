package semanticweb.sparql.utils;

import weka.core.Instances;
import weka.filters.unsupervised.attribute.Standardize;

public class AttributeFilterMeta {
	private Instances instances;
	private int[] removedAttributes;
	private Standardize standardizeFilter;
	private String classAtrributeValues;
	public Instances getInstances() {
		return instances;
	}
	public void setInstances(Instances instances) {
		this.instances = instances;
	}
	public int[] getRemovedAttributes() {
		return removedAttributes;
	}
	public void setRemovedAttributes(int[] removedAttributes) {
		this.removedAttributes = removedAttributes;
	}
	public Standardize getStandardizeFilter() {
		return standardizeFilter;
	}
	public void setStandardizeFilter(Standardize standardizeFilter) {
		this.standardizeFilter = standardizeFilter;
	}
	public String getClassAtrributeValues() {
		return classAtrributeValues;
	}
	public void setClassAtrributeValues(String classAtrributeValues) {
		this.classAtrributeValues = classAtrributeValues;
	}
	
}

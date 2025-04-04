package test;

import org.apache.commons.math3.stat.correlation.PearsonsCorrelation;

import semanticweb.sparql.utils.StatUtils;

import com.hp.hpl.jena.sparql.sse.Tags;

import weka.classifiers.functions.LibSVM;
import weka.core.SelectedTag;
import weka.core.Tag;

public class LibTests {
	
	public static void rSquareTest(){
		
		PearsonsCorrelation pc = new PearsonsCorrelation();
		double[] y_true = {1.0, 1.0, 5.0, 2.0, 1.0, 1.0, 1.0, 3.0, 1.0, 3.0, 1.0, 4.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
		double[] y_pred = {5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0};
		
		double pcVal = pc.correlation(y_true, y_pred);
		
		//only valid for linear regression
		double rSquared = pcVal * pcVal;
		System.out.println(rSquared);
		
		System.out.println(StatUtils.rSquared(y_true, y_pred));		
	}

	/**
	 * @param args
	 */
	public static void main(String[] args) {

//
//		LibSVM svr = new LibSVM();
//		//System.out.println(LibSVM.TAGS_SVMTYPE);
//		
//		for(Tag tag:LibSVM.TAGS_SVMTYPE) {
//			System.out.println(tag);
//		}
//		
//		SelectedTag nuSVR = new SelectedTag(LibSVM.SVMTYPE_NU_SVR, LibSVM.TAGS_SVMTYPE);
		
		
		rSquareTest();
		
		
	}

}

package semanticweb.sparql.utils;

import java.util.HashMap;
import java.util.Map;

public class PredHistogram {
    private String onSubType,onObjType, onSubQuery, onObjQuery, predicate;

    private HashMap<String, Double> valuesSub;
    private HashMap<String, Double> valuesObj;
    public PredHistogram(String type, String on, String query, String predicate) {

        if (on.equals("sub")) {
            this.onSubQuery = query;
            this.onSubType = type;
        }
        else {
            this.onObjQuery = query;
            this.onObjType = type;
        }
        this.predicate = predicate;
    }
    public PredHistogram(String type, String on, HashMap<String, Double> values, String predicate) {
        if (on.equals("sub")) {
            this.valuesSub = values;
            this.onSubType = type;
        }
        else {
            this.valuesObj = values;
            this.onObjType = type;
        }
        this.predicate = predicate;
    }
    public void setValuesHist(String on, String type, HashMap<String, Double> values){
        if (on.equals("sub")) {
            this.valuesSub = values;
            this.onSubType = type;
        }
        else {
            this.valuesObj = values;
            this.onObjType = type;
        }
    }

    public String getOnSubType() {
        return onSubType;
    }

    public String getOnObjType() {
        return onObjType;
    }

    public String getOnSubQuery() {
        return onSubQuery;
    }

    public String getOnObjQuery() {
        return onObjQuery;
    }

    public String getPredicate() {
        return predicate;
    }

    private String printSubQuery(String output_element_delimiter) {

        return this.predicate +
                output_element_delimiter +
                this.onSubType +
                output_element_delimiter +
                "sub" +
                output_element_delimiter +
                this.onSubQuery.replaceAll("\n"," ");
    }
    private String printObjQuery(String output_element_delimiter) {
        return this.predicate +
                output_element_delimiter +
                this.onObjType +
                output_element_delimiter +
                "obj" +
                output_element_delimiter +
                this.onObjQuery.replaceAll("\n"," ");
    }

    public void setQuery(String on,String type, String query) {
        if (on.equals("sub")) {
            this.onSubQuery = query;
            this.onSubType = type;
        }
        else {
            this.onObjQuery = query;
            this.onObjType = type;
        }
    }

    public String toSamplingQueryFileString(String output_element_delimiter) {
        if (this.onObjQuery != null && this.onSubQuery != null)
            return printSubQuery(output_element_delimiter) + "\n" + printObjQuery(output_element_delimiter);
        if (this.onObjQuery != null){
            return printObjQuery(output_element_delimiter);
        }
        return printSubQuery(output_element_delimiter);
    }

    public String printSubHistDataFileString(String output_delimiter, String output_element_delimiter) {
        StringBuilder sb = new StringBuilder();
        // Add some bonus marks
        // to all the students and print it
        sb.append(this.predicate);
        sb.append(output_delimiter);
        sb.append(this.onSubType);
        sb.append(output_delimiter);
        sb.append("sub");
        sb.append(output_delimiter);
        for (String keyVal: this.valuesSub.keySet().toArray(new String[0])) {
            double valVal = this.valuesSub.get(keyVal);
            sb.append(keyVal);
            sb.append(output_element_delimiter);
            sb.append(valVal);
            sb.append(output_element_delimiter);
        }
        sb.append("\n");
        return  sb.toString();
    }
    public String printObjHistDataFileString(String output_delimiter, String output_element_delimiter) {
        StringBuilder sb = new StringBuilder();
        // Add some bonus marks
        // to all the students and print it
        sb.append(this.predicate);
        sb.append(output_delimiter);
        sb.append(this.onObjType);
        sb.append(output_delimiter);
        sb.append("obj");
        sb.append(output_delimiter);
        for (String keyVal: this.valuesObj.keySet().toArray(new String[0])) {
            double valVal = this.valuesObj.get(keyVal);
            sb.append(keyVal);
            sb.append(output_element_delimiter);
            sb.append(valVal);
            sb.append(output_element_delimiter);
        }
        sb.append("\n");
        return  sb.toString();
    }
    public String printHistDataFileString(String output_delimiter, String output_element_delimiter) {
        if (this.onObjType != null && this.onSubType != null)
            return printSubHistDataFileString(output_delimiter, output_element_delimiter) + "\n" + printObjHistDataFileString(output_delimiter, output_element_delimiter);
        if (this.onObjType != null){
            return printObjHistDataFileString(output_delimiter, output_element_delimiter);
        }
        return printSubHistDataFileString(output_delimiter, output_element_delimiter);
    }
}

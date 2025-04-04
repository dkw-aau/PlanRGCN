package semanticweb.sparql.utils;

import java.util.HashMap;
import java.util.Map;

/**
 * This class represent the sampling needed for a predicate.
 */
public class Sample {

    private String pred;
    private String type;
    private String operator;
    private String value;
    private HashMap<String,Object> histData;
    /**
     * Contructor
     * @param pred String predicate name
     * @param type The type could be [number,uri,literal], usefull to select the to_string behaviour to output
     */
    public Sample(String pred, String type,String operator, String value) {
        this.pred = pred;
        this.type = type;
        this.value = type;
        this.operator = operator;
    }

    public String getPred() {
        return pred;
    }

    public void setPred(String pred) {
        this.pred = pred;
    }

    public String getType() {
        return type;
    }

    public void setType(String type) {
        this.type = type;
    }

    public HashMap<String, Object> getHistData() {
        return histData;
    }

    public void setHistData(HashMap<String, Object> histData) {
        this.histData = histData;
    }
    //Todo
    private String getEncoding() {
        if (this.type.equals("number")){
            int minValue = (int) this.histData.get("minValue");
            int maxValue = (int) this.histData.get("maxValue");
            int distinctValues = (int) this.histData.get("maxValue");
        }
        return "";
    }
}

package lsq_extract.util;

import java.util.HashMap;

public class Query {
    public String id;
    public String text;
    public String timestamp;
    public double projectVars;
    public double tripleCount;
    public double joinVertexCount;
    public int resultSize = -1;
    public double latency = -1;
    public double duration;
    HashMap<String,Double> operators;
    
    public Query(String text){
        this.text = text;
        this.operators = new HashMap<>();
    }
    public Query(String id, String text, String timestamp, 
    double projectVars, double tripleCount, double joinVertex, double duration){
        this.id = id;
        this.text = text;
        this.timestamp = timestamp;
        this.projectVars = projectVars;
        this.tripleCount = tripleCount;
        this.joinVertexCount = joinVertex;
        this.duration = duration;
        this.operators = new HashMap<>();
    }
    public Query(String id, String text, double duration){
        this.id = id;
        this.text = text;
        this.duration = duration;
        this.operators = new HashMap<>();
    }

    public int addOp(String key, double value){
        this.operators.put(key, value);
        return 1;
    }

    //-1 means that the operator does not exist. This should not happen.
    public double getOp(String key){
        Double t = this.operators.get(key);
        if (t == null){
            return 0;
        }
        return t;
    }
}

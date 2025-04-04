package Benchmark;

public class Query {
    String id;
    String queryString;
    String type;
    int noNotedTriplePatterns;
    int noTrplPatterns;
    double runTimeMS;
    int resultSize;

    public Query() {
    }

    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public String getQueryString() {
        return queryString;
    }

    public void setQueryString(String queryString) {
        this.queryString = queryString;
    }

    public String getType() {
        return type;
    }

    public void setType(String type) {
        this.type = type;
    }

    public int getNoNotedTriplePatterns() {
        return noNotedTriplePatterns;
    }

    public void setNoNotedTriplePatterns(int noNotedTriplePatterns) {
        this.noNotedTriplePatterns = noNotedTriplePatterns;
    }

    public int getNoTrplPatterns() {
        return noTrplPatterns;
    }

    public void setNoTrplPatterns(int noTrplPatterns) {
        this.noTrplPatterns = noTrplPatterns;
    }

    public double getRunTimeMS() {
        return runTimeMS;
    }

    public void setRunTimeMS(double runTimeMS) {
        this.runTimeMS = runTimeMS;
    }

    public int getResultSize() {
        return resultSize;
    }

    public void setResultSize(int resultSize) {
        this.resultSize = resultSize;
    }
    //s,query,type,meanJoinVerticesDegree,bgps,joinVertices,triplePatterns,runTimeMs,resultSize
}

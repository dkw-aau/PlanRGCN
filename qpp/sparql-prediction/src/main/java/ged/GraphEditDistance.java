package ged;

import Benchmark.Query;

public class GraphEditDistance{

    public double distance(Query q1, Query q2) throws Exception {
        RDFGraphMatching matcher = new RDFGraphMatching();
        return matcher.queryGraphDistance(q1.getQueryString(),q2.getQueryString());
    }

    public double distance(String q1, String q2) throws Exception {
        RDFGraphMatching matcher = new RDFGraphMatching();
        return matcher.queryGraphDistance(q1,q2);
    }
}

package lsq_extract.lsq_execution;

import java.io.IOException;
import java.util.ArrayList;



public class LSQExecutorDuration extends LSQExecutor{
    ArrayList<ArrayList<Integer>> intervals;
    
    public LSQExecutorDuration(String serviceURL, int limit){
        super(serviceURL, limit);
    }

    public LSQExecutorDuration(String serviceURL, int limit,ArrayList<ArrayList<Integer>> intervals){
        super(serviceURL, limit);
        this.intervals = intervals;
    }

    public void extractQueries(String path){
        double lower = 0.0;
        double higher = 0.01;
        try {
            String query = getQueryString(lower, higher);
            execSelectAndSave(path, query, true);

            lower = 0.01;
            higher = 0.1;
            query = getQueryString(lower, higher);
            execSelectAndSave(path, query, false);

            lower = 0.1;
            higher = 77546.44436472;
            query = getQueryString(lower, higher);
            execSelectAndSave(path, query, false);
        } catch (IOException e) {
            e.printStackTrace();
        }

        
    }

    public void extractRefinedIntervalQueries(String path){
        double lower = 0.10103753;
        double higher = 1;
        try {
            String query = getQueryString(lower, higher);
            execSelectAndSave(path, query, true);

            lower = 1;
            higher = 10;
            query = getQueryString(lower, higher);
            execSelectAndSave(path, query, false);

            lower = 10;
            higher = 100;
            query = getQueryString(lower, higher);
            execSelectAndSave(path, query, false);

            lower = 100;
            higher = 77546.44436472;
            query = getQueryString(lower, higher);
            execSelectAndSave(path, query, false);
        } catch (IOException e) {
            e.printStackTrace();
        }

        
    
    }

    public String getQueryStringv2(double lower,double higher){
        return """
            PREFIX lsqv:   <http://lsq.aksw.org/vocab#>  
            SELECT (SAMPLE(?timestamp2) AS ?timestamp) ?query (SAMPLE(?queryString2) AS ?queryString) (SAMPLE(?projects2) AS ?projects) (SAMPLE(?triples2) AS ?triples) (SAMPLE(?joinVertices2) AS ?joinVertices) (SAMPLE(?duration2) AS ?duration)
            FROM <http://www.purl.com/Dbpedia>
                WHERE {
                    ?query      lsqv:hasRemoteExec ?queryRemoteExec2 .
                    ?query      lsqv:hasLocalExec ?queryLocalExec .
                    ?queryRemoteExec2  <http://www.w3.org/ns/prov#atTime> ?timestamp2 .
                    ?queryLocalExec2 lsqv:hasQueryExec ?executionInfo2 .
                    ?executionInfo2 lsqv:evalDuration ?duration2 .
                  
                    ?query      lsqv:text ?queryString2 ;
                                lsqv:hasStructuralFeatures ?feats2 .
                    ?feats2      
                                lsqv:projectVarCount ?projects2;
                                lsqv:tpCount ?triples2;
                                lsqv:joinVertexCount ?joinVertices2 .
                                FILTER("""+lower+" < ?duration2 && ?duration2 < "+higher+")} GROUP BY ?query LIMIT " + this.limit;
    }


    public String getQueryString(double lower,double higher){
        return """
            PREFIX lsqv:   <http://lsq.aksw.org/vocab#>  
            SELECT ?timestamp ?query ?queryString ?projects ?triples ?joinVertices ?duration
            FROM <http://www.purl.com/Dbpedia>
                WHERE {
                    ?query      lsqv:hasRemoteExec ?queryRemoteExec .
                    ?query      lsqv:hasLocalExec ?queryLocalExec .
                    ?queryRemoteExec  <http://www.w3.org/ns/prov#atTime> ?timestamp .
                    ?queryLocalExec lsqv:hasQueryExec ?executionInfo .
                    ?executionInfo lsqv:evalDuration ?duration .
                  
                    ?query      lsqv:text ?queryString ;
                                lsqv:hasStructuralFeatures ?feats .
                    ?feats      
                                lsqv:projectVarCount ?projects;
                                lsqv:tpCount ?triples;
                                lsqv:joinVertexCount ?joinVertices .
                                FILTER("""+lower+" < ?duration && ?duration < "+higher+")} LIMIT " + this.limit;
    }
}
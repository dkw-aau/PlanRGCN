package lsq_extract.benchmark;

import java.util.ArrayList;
import java.util.concurrent.TimeUnit;


import org.apache.jena.ext.com.google.common.base.Stopwatch;
import org.apache.jena.query.ARQ;
import org.apache.jena.query.Query;
import org.apache.jena.query.QueryExecution;
import org.apache.jena.query.QueryExecutionFactory;
import org.apache.jena.query.QueryFactory;
import org.apache.jena.query.ResultSet;

public class QueryExecutor {
    //public static String serviceURL = "http://172.21.233.23:81/Dbpedia/query";
    String serviceURL = "http://172.21.233.23:81/Dbpedia/query";
    ArrayList<lsq_extract.util.Query> queries;
    Stopwatch watch = Stopwatch.createUnstarted();
    public QueryExecutor(String URL, ArrayList<lsq_extract.util.Query> qs){
        this.serviceURL = URL;
        this.queries = qs;
        ARQ.init();
    }
    public void benchmarkQueries(){
        for(lsq_extract.util.Query i : this.queries){
            this.executeQuery(i);
        }
    }
    public void executeQuery(lsq_extract.util.Query query){
        Query q = QueryFactory.create(query.text);
        this.watch.reset();
        this.watch.start();
        QueryExecution qe = QueryExecutionFactory.sparqlService(this.serviceURL,
                q);
        ResultSet results = qe.execSelect();
        
        int resultSize = 0;
        while (results.hasNext()) {
            resultSize++;
            results.nextSolution();
        }
        double elapsed = this.watch.elapsed(TimeUnit.NANOSECONDS);
        query.resultSize = resultSize;
        query.latency = elapsed;
    }

}

package lsq_extract.lsq_execution;

import java.io.IOException;
import java.util.ArrayList;

import org.apache.jena.query.QuerySolution;

import lsq_extract.util.LSQDataWriterWStats;

public class LSQExcutorDurationV2 extends LSQExecutorDuration{

    public LSQExcutorDurationV2(String serviceURL, int limit) {
        super(serviceURL, limit);
        LSQDataWriterWStats.header = new String[]{"queryID","queryString", "duration"};
        
    }

    public LSQExcutorDurationV2(String serviceURL, int limit,ArrayList<ArrayList<Integer>> intervals){
        super(serviceURL, limit, intervals);
    }

    @Override
    public String getQueryString(double lower,double higher){
        return """
            PREFIX lsqv:   <http://lsq.aksw.org/vocab#>  
            SELECT ?query (SAMPLE(?queryString2) AS ?queryString) (SAMPLE(?duration2) AS ?duration)
            FROM <http://www.purl.com/Dbpedia>
                WHERE {
                    ?query      lsqv:hasRemoteExec ?queryRemoteExec .
                    ?query      lsqv:hasLocalExec ?queryLocalExec .
                    ?queryLocalExec lsqv:hasQueryExec ?executionInfo .
                    ?executionInfo lsqv:evalDuration ?duration2 .
                    ?query      lsqv:text ?queryString2 .
             
              FILTER("""+lower+ " < ?duration2 && ?duration2 < "+higher+")} GROUP BY ?query LIMIT "+ this.limit;
    }

    @Override
    public void writeData(QuerySolution soln, LSQDataWriterWStats writer) throws IOException{
        String lsqString = String.valueOf( soln.get("queryString"));
        //Query can contain some space manipulation characters that are best to avoid when logging to csv.
        lsqString = lsqString.replaceAll("\t", " ").replaceAll("\r"," ").replaceAll("\n"," ");
        lsqString = lsqString.replaceAll("\s+", " ");
            
        String queryId = String.valueOf(soln.get("query"));
        String duration = String.valueOf(soln.getLiteral("duration").getDouble());
        writer.writeCSV(queryId, lsqString, duration);
    }
}

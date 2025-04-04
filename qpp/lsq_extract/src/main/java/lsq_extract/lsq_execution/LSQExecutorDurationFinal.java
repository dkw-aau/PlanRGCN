package lsq_extract.lsq_execution;

import java.io.IOException;


import org.apache.jena.query.ARQ;
import org.apache.jena.query.QuerySolution;

import lsq_extract.util.LSQDataWriterWStats;

public class LSQExecutorDurationFinal extends LSQExcutorDurationV2{
    public LSQExecutorDurationFinal(String serviceURL, int limit) {
        super(serviceURL, limit);
        LSQDataWriterWStats.header = new String[]{"queryID","queryString", "duration", "resultCount"};
        ARQ.init();
    }

    public void extractQueries(String path){
        double lower = 0.0;
        double higher = 0.01; //8000
        try {
            String query = getQueryString(lower, higher);
            execSelectAndSave(path, query, true);

            lower = 0.01;
            higher = 0.1; //7896
            query = getQueryString(lower, higher);
            execSelectAndSave(path, query, false);
            
            lower = 0.1;
            higher = 1; //7089
            query = getQueryString(lower, higher);
            execSelectAndSave(path, query, false);

            lower = 1;
            higher = 10; // 7996     8953 select queries in 2015-2016 dbpedia in total
            query = getQueryString(lower, higher);
            execSelectAndSave(path, query, false);

            lower = 10;
            higher = 100;//7964    9080 select queriees in 2015-2016 dbpedia in total
            query = getQueryString(lower, higher);
            execSelectAndSave(path, query, false);
            lower = 100;
                //higher = 77546.44436472;
            higher = 84712; //21 queries   30 queires in total including describe.
            query = getQueryString(lower, higher);
                execSelectAndSave(path, query, false);
            
        } catch (IOException e) {
            e.printStackTrace();
        }

        
    
    }

    public void execute_longest_interval(String path){
        try {
                double lower = 100;
                //higher = 77546.44436472;
                double higher = 84712;
                String query = getQueryString(lower, higher);
                execSelectAndSave(path, query, false);
        } catch (IOException e) {
                // TODO Auto-generated catch block
                e.printStackTrace();
        }
    }

    @Override
    public String getQueryString(double lower,double higher){
        return """
            PREFIX lsqv:   <http://lsq.aksw.org/vocab#>  
            SELECT ?query (SAMPLE(?queryString2) as ?queryString) (SAMPLE(?duration2) as ?duration) (SAMPLE(?rc2) as ?rc)
                                        WHERE {
                                            ?query      lsqv:hasLocalExec ?queryLocalExec .
                                            ?queryLocalExec lsqv:hasQueryExec ?executionInfo .
                                            ?executionInfo lsqv:evalDuration ?duration2 .
                                            
                                            ?query      lsqv:text ?queryString2 .
                                              ?query lsqv:hasSpin ?spin .
                                              ?spin a <http://spinrdf.org/sp#Select> . 
                                              OPTIONAL {
                                                          ?executionInfo lsqv:resultCount ?rc2 .}
                                     
                                                          FILTER("""+Double.toString(lower)+ "  < ?duration2 && ?duration2 < " + Double.toString(higher)+")} GROUPBY ?query LIMIT " + this.limit;
    }

    public String getQueryStringLongest(double lower,double higher){
        return """
            PREFIX lsqv:   <http://lsq.aksw.org/vocab#>  
            SELECT ?query (?queryString2 as ?queryString) (?duration2 as ?duration) (?rc2 as ?rc)
                                        WHERE {
                                            ?query      lsqv:hasLocalExec ?queryLocalExec .
                                            ?queryLocalExec lsqv:hasQueryExec ?executionInfo .
                                            ?executionInfo lsqv:evalDuration ?duration2 .
                                            
                                            ?query      lsqv:text ?queryString2 .
                                              ?query lsqv:hasSpin ?spin .
                                              ?spin a <http://spinrdf.org/sp#Select> . 
                                              OPTIONAL {
                                                          ?executionInfo lsqv:resultCount ?rc2 .}
                                     
                                      FILTER(100  < ?duration2 && ?duration2 < 84712)}  LIMIT 8000""";
    }
            
            
    public String getQueryString_has_duplicated_durations(double lower,double higher){
        return """
            PREFIX lsqv:   <http://lsq.aksw.org/vocab#>  
            SELECT distinct ?query ?queryString2 ?duration2 ?rc
                            WHERE {
                                ?query      lsqv:hasLocalExec ?queryLocalExec .
                                ?queryLocalExec lsqv:hasQueryExec ?executionInfo .
                                ?executionInfo lsqv:evalDuration ?duration2 .
                                
                                ?query      lsqv:text ?queryString2 .
                                  ?query lsqv:hasSpin ?spin .
                                  ?spin a <http://spinrdf.org/sp#Select> . 
                                  OPTIONAL {
                                              ?executionInfo lsqv:resultCount ?rc .}
                         
                          FILTER("""+Double.toString(lower)+ "  < ?duration2 && ?duration2 < " + Double.toString(higher)+")} LIMIT " + this.limit;
    }
    

    @Override
    public void writeData(QuerySolution soln, LSQDataWriterWStats writer) throws IOException{
        String lsqString = String.valueOf( soln.get("queryString"));
        //Query can contain some space manipulation characters that are best to avoid when logging to csv.
        lsqString = lsqString.replaceAll("\t", " ").replaceAll("\r"," ").replaceAll("\n"," ");
        lsqString = lsqString.replaceAll("\s+", " ");
            
        String queryId = String.valueOf(soln.get("query"));
        String duration = String.valueOf(soln.getLiteral("duration").getDouble());
        String rc = "";
        if(soln.get("rc") != null)
            rc = String.valueOf(soln.getLiteral("rc").getLong());
        //{"queryID","queryString", "duration", "resultCount"}
        writer.writeCSV(queryId, lsqString, duration, rc);
    }
}

/*
 * PREFIX lsqv:   <http://lsq.aksw.org/vocab#>  
SELECT distinct ?query ?queryString2 ?duration2 ?rc
                WHERE {
                    ?query      lsqv:hasLocalExec ?queryLocalExec .
                    ?queryLocalExec lsqv:hasQueryExec ?executionInfo .
                    ?executionInfo lsqv:evalDuration ?duration2 .
					
                    ?query      lsqv:text ?queryString2 .
  					#?query lsqv:hasSpin ?spin .
      				#?spin a <http://spinrdf.org/sp#Select> . 
  					OPTIONAL {
  								?executionInfo lsqv:resultCount ?rc .}
             
              FILTER(100  < ?duration2 && ?duration2 < 847112)} LIMIT 30
 * 
*/
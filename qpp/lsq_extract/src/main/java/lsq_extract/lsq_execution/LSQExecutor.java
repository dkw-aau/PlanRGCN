package lsq_extract.lsq_execution;

import java.io.IOException;

import org.apache.jena.query.ARQ;
import org.apache.jena.query.Query;
import org.apache.jena.query.QueryExecution;
import org.apache.jena.query.QueryExecutionFactory;
import org.apache.jena.query.QueryFactory;
import org.apache.jena.query.QuerySolution;
import org.apache.jena.query.ResultSet;
import org.apache.jena.sparql.engine.http.QueryEngineHTTP;

import lsq_extract.util.LSQDataWriterWStats;

public class LSQExecutor extends LSQExecutorBase{
    int limit = 1000000;
    int offset = 0;
    String queryString2;
    public LSQExecutor(String serviceURL, int limit){
        super();
        this.serviceURL = serviceURL;
        this.limit = limit;
        //for virtuoso
        //this.serviceURL = "http://172.21.233.23:8890/sparql";
        
        //The following would work but time out can happen if public endpoint is used.
        this.queryString = """
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
            
            } LIMIT """ + this.limit
            //+"ORDER BY DESC (?timestamp)"+
            //"OFFSET "+ this.offset +" LIMIT" +  
            //+ this.limit 
            ;
        this.queryString2 = """
            PREFIX lsqv:   <http://lsq.aksw.org/vocab#>  
            SELECT
                """;

    }
    //feats lsqv:bpgCount ?bgpCount ; apparently does not exist even through it is presented in the paper.
    
    @Override
    public void extractLSQQueryLog(String outputPath){
        
        try {
            this.execSelectAndSave(
                    outputPath,
                    queryString);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    //@Override
    public void execSelectAndSave(String output_path, String queryString, Boolean writerHeader) throws IOException {
        System.out.println("here");
        Query query = QueryFactory.create(queryString) ;
        System.out.println("here2");
        QueryExecution q = QueryExecutionFactory.sparqlService(this.serviceURL,
                query);
        System.out.println(this.serviceURL);
        ResultSet results = q.execSelect();
        System.out.println("here4");
        LSQDataWriterWStats writer;
        if (writerHeader == null || writerHeader==true){
            writer = new LSQDataWriterWStats(output_path, true);
        }else{
            writer = new LSQDataWriterWStats(output_path, false);
        }
        System.out.println("here5");
        while (results.hasNext()) {
            QuerySolution soln = results.nextSolution();
            writeData(soln, writer);
        }
        writer.close();
    }
    public void writeData(QuerySolution soln, LSQDataWriterWStats writer) throws IOException{
        String lsqString = String.valueOf( soln.get("queryString"));
        //Query can contain some space manipulation characters that are best to avoid when logging to csv.
        lsqString = lsqString.replaceAll("\t", " ").replaceAll("\r"," ").replaceAll("\n"," ");
        
        //lsqString = lsqString.replaceAll("\s+", " ");
        if (lsqString.contains("§")){
                return;
        }
        String timestamp = String.valueOf(soln.get("timestamp"));
            
        String queryId = String.valueOf(soln.get("query"));
        String projectVars = String.valueOf(soln.getLiteral("projects").getInt());
        String tripleCount = String.valueOf(soln.getLiteral("triples").getInt());
        String joinVertices = String.valueOf(soln.getLiteral("joinVertices").getInt());
        String duration = String.valueOf(soln.getLiteral("duration").getDouble());

        
        
        writer.writeCSV(queryId,timestamp, lsqString, projectVars,tripleCount,joinVertices, duration);
    }
    public void extractLSQQueryLogNotIn(String outputPath){
        
        try {
            this.execSelectAndSave(
                    outputPath,
                    queryString);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}

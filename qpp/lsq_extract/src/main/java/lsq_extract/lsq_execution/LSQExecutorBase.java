package lsq_extract.lsq_execution;
import java.io.IOException;

import org.apache.jena.query.ARQ;
import org.apache.jena.query.Query;
import org.apache.jena.query.QueryExecution;
import org.apache.jena.query.QueryExecutionFactory;
import org.apache.jena.query.QueryFactory;
import org.apache.jena.query.QuerySolution;
import org.apache.jena.query.ResultSet;

import lsq_extract.util.LSQDataWriter;

public class LSQExecutorBase {
    public String serviceURL = "http://172.21.233.23:81/DBpedia/query";
    String queryString;

    public LSQExecutorBase(){
        queryString = "SELECT ?timestamp ?query ?queryString\n" +
                "WHERE {\n" +
                "  ?queryExec <http://www.w3.org/ns/prov#atTime> ?timestamp .\n" +
                "  #?query <http://lsq.aksw.org/vocab#hasQueryExec> ?queryExec .\n" +
                "  ?query <http://lsq.aksw.org/vocab#hasRemoteExec> ?queryExec .\n" +
                "  ?query <http://lsq.aksw.org/vocab#text> ?queryString .\n" +
                "\n" +
                "}\n" +
                "ORDER BY DESC (?timestamp)";
    }

    public void extractLSQQueryLog(String outputPath){
        
        try {
            this.execSelectAndSave(
                    outputPath,
                    queryString);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public void execSelectAndSave(String output_path, String queryString) throws IOException {
        ARQ.init();
        Query query = QueryFactory.create(queryString) ;
        QueryExecution q = QueryExecutionFactory.sparqlService(this.serviceURL,
                query);
        ResultSet results = q.execSelect();

        LSQDataWriter writer = new LSQDataWriter(output_path);//"/Users/abirammohanaraj/Documents/GitHub/lsq_extract/lsq_data.csv"
        //ResultSetFormatter.out(System.out, results);

        while (results.hasNext()) {
            QuerySolution soln = results.nextSolution();
            String lsqString = String.valueOf( soln.get("queryString"));
            lsqString = lsqString.replaceAll("\t", " ").replaceAll("\r"," ").replaceAll("\n"," ");
            String timestamp = String.valueOf(soln.get("timestamp"));
            String queryId = String.valueOf(soln.get("query"));
            writer.writeCSV(queryId,timestamp,lsqString);
        }
        writer.close();
    }
}

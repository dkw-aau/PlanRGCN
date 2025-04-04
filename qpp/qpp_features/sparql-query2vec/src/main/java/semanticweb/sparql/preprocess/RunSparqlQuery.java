package semanticweb.sparql.preprocess;

import org.apache.jena.query.*;
import org.apache.jena.rdf.model.Model;

import java.util.ArrayList;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.TimeUnit;

public class RunSparqlQuery implements Callable<ArrayList<String>> {
    private final Map.Entry<String, Query> entry;
    private Model triples;
    private String endpoint;
    public RunSparqlQuery(Model triples, Map.Entry<String, Query> entry) {
        this.entry = entry;
        this.triples = triples;
    }

    public RunSparqlQuery(String endpoint, Map.Entry<String, Query> entry) {
        this.entry = entry;
        this.endpoint = endpoint;
    }

    @Override
    public ArrayList<String> call() throws Exception
    {
        ArrayList<String> result = new ArrayList<>();
        QueryExecution qexec = QueryExecutionFactory.sparqlService(endpoint, entry.getValue());
        qexec.setTimeout(90, TimeUnit.SECONDS);
        try {
//            QueryExecution qexec = QueryExecutionFactory.create(entry.getValue(), triples);

            long start = System.currentTimeMillis();
            ResultSet results = qexec.execSelect() ;
            long end = System.currentTimeMillis();
            long elapsedTime = end - start;
            int count = 0;
            while (results.hasNext()){
                results.next();
                count++;
            }
            System.out.println("Completed query");
            result.add(entry.getKey()); //id query
            result.add(entry.getValue().toString().replaceAll("\n"," ")); //query
            result.add(String.valueOf(count)); //cardinality
            result.add(String.valueOf(elapsedTime)); //time in miliseconds
            }
            catch (Exception e) {
                e.printStackTrace();
            }
            finally {
                qexec.close();
            }

        return result;
    }
}

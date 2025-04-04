package semanticweb.sparql.preprocess;

import org.apache.jena.query.*;
import org.apache.jena.system.Txn;

import java.util.ArrayList;
import java.util.concurrent.*;

import org.apache.jena.dboe.base.file.Location;
import org.apache.jena.tdb2.TDB2Factory;

public class RunSparqlQuery2 {
    //    private final Dataset dataset;
    private String id;
    private final String outputDelimiter;
    private final Query query;
    private final Location dataset;

    public RunSparqlQuery2(Location dataset, String id, Query query, String outputDelimiter) {
        this.query = query;
        this.dataset = dataset;
        this.id = id;
        this.outputDelimiter = outputDelimiter;
    }

    public String getData() {
        ArrayList<String> result = new ArrayList<>();
        Dataset ds = TDB2Factory.connectDataset(this.dataset);
        Txn.executeRead(ds, ()-> {
        int count = 0;
        try (QueryExecution qexec = QueryExecutionFactory.create(query, ds)) {

            qexec.setTimeout(60, TimeUnit.SECONDS, 300, TimeUnit.SECONDS);


            long start = System.currentTimeMillis();
            ResultSet results = qexec.execSelect();

            System.out.println("Starting to iterate over results");
            while (results.hasNext()) {
                results.next();
                count++;
            }
            long end = System.currentTimeMillis();
            long elapsedTime = end - start;
            System.out.println("Completed query".concat(String.valueOf(count)).concat("results"));
            result.add(id); //id query
            result.add(query.toString().replaceAll("\n", " ")); //query
            result.add(String.valueOf(count)); //cardinality
            result.add(String.valueOf(elapsedTime));
            qexec.close();
        } catch (QueryCancelledException exception) {
            System.err.println("Query Cancelada: ".concat(id).concat(" ").concat(exception.toString()));
            result.add(id); //id query
            result.add(query.toString().replaceAll("\n", " ")); //query
            result.add(String.valueOf(count)); //cardinality
            result.add(String.valueOf(-1));
        } catch (Exception e) {
            System.err.println("Exception: ".concat(e.toString()));
        } finally {
            ds.close();
        }
        });

        return result.get(0)//id query
                .concat(outputDelimiter)
                .concat(result.get(1)) //query
                .concat(outputDelimiter)
                .concat(result.get(2))
                .concat(outputDelimiter)//cardinality
                .concat(result.get(3)); //time in miliseconds
    }

    public String call() {
        ExecutorService executor = Executors.newCachedThreadPool();
        Callable<String> task = new Callable<String>() {
            public String call() {
                return getData();
            }
        };
        Future<String> future = executor.submit(task);
        String result = "";
        try {
            result = future.get(300, TimeUnit.SECONDS);
        } catch (TimeoutException ex) {
            // handle the timeout
            System.err.println("Query timeout: ".concat(id));
            return id
                    .concat(outputDelimiter)//id query
                    .concat(query.toString().replaceAll("\n", " ")) //query
                    .concat(outputDelimiter)
                    .concat(String.valueOf(0))
                    .concat(outputDelimiter)//cardinality
                    .concat(String.valueOf(-1));
        } catch (InterruptedException e) {
            // handle the interrupts
            System.err.println("Query InterruptedException: ".concat(id).concat(" ".concat(e.toString())));
        } catch (ExecutionException e) {
            // handle other exceptions
            System.err.println("Query ExecutionException: ".concat(id).concat(" ".concat(e.toString())));
            return id
                    .concat(outputDelimiter)//id query
                    .concat(query.toString().replaceAll("\n", " ")) //query
                    .concat(outputDelimiter)
                    .concat(String.valueOf(0))
                    .concat(outputDelimiter)//cardinality
                    .concat(String.valueOf(-1));
        } finally {
            future.cancel(true); // may or may not desire this
        }
        System.out.println(result);
        return result;
    }
}

package semanticweb.sparql.preprocess;

import liquibase.util.csv.opencsv.CSVReader;
import org.apache.commons.codec.digest.DigestUtils;
import org.apache.jena.dboe.base.file.Location;
import org.apache.jena.query.*;
import org.apache.jena.rdf.model.Model;
import org.apache.jena.rdf.model.ModelFactory;
import org.apache.jena.sparql.algebra.Algebra;
import org.apache.jena.sys.JenaSystem;
import org.apache.log4j.LogManager;
import org.apache.log4j.Logger;
import semanticweb.sparql.SparqlUtils;

import java.io.*;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.util.*;

/**
 *
 * @author cbuil
 */
public class RunQueriesSequential
{

    private static Logger logger = LogManager
            .getLogger("runParallelQueries");

    private String queryString;
    private String inputLogsFile;
    private  Location dataset;
    private String  outputDelimiter;
    private String  namespaces;
    private String outputFile = "";
    private int init_row = 0;
    private int len_exec_results;
    private static boolean evaluation = false;
    private int idColumn, queryColumn;

    public RunQueriesSequential(String inputLogsFile, String inputTDB2Url, String outputFile, String namespaces, int init_row, int len_exec_results, String outputDelimiter, int idColumn, int queryColumn) {
        this.inputLogsFile = inputLogsFile;
//        JenaSystem.init();
//        ARQ.init();
        this.dataset  = Location.create(inputTDB2Url);
        this.outputFile = outputFile;
        this.namespaces = namespaces;
        this.outputDelimiter = outputDelimiter;
        this.idColumn = idColumn;
        this.queryColumn = queryColumn;
        this.init_row = init_row;
        if(len_exec_results == -1){
            //
            this.len_exec_results = Integer.MAX_VALUE;
        }
        else this.len_exec_results = len_exec_results;
    }
    public RunQueriesSequential(String inputLogsFile, String inputTDB2Url, String outputFile, String namespaces, int init_row, int len_exec_results, String outputDelimiter) {
        this.inputLogsFile = inputLogsFile;
        JenaSystem.init();
        ARQ.init();
        this.dataset = Location.create(inputTDB2Url);
//        this.dataset = TDB2Factory.connectDataset(location);
        this.outputFile = outputFile;
        this.namespaces = namespaces;
        this.init_row = init_row;
        if(len_exec_results == -1){
            //
            this.len_exec_results = Integer.MAX_VALUE;
        }
        else this.len_exec_results = len_exec_results;
        this.outputDelimiter = outputDelimiter;
    }

    /*public HashMap<String,Query> getQueries(int  idColumn, int queryColumn, char input_delimiter){
        System.out.println("Reading Queries..");
        boolean header = true;
        HashMap<String,Query> queries = new HashMap<>();
        int index = 0;

        int len = len_exec_results;
        try {
            InputStreamReader csv = new InputStreamReader(new FileInputStream(inputLogsFile));
            CSVReader csvReader = new CSVReader (csv, input_delimiter);
            String[] record;

            while ((record = csvReader.readNext()) != null && len > 0) {
                index++;
                if (header || index < init_row){
                    header = false;
                    continue;
                }

                String query = record[queryColumn];
//                String query = record[queryColumn].replaceAll("^\"|\"$", "");
                String id = record[idColumn];
                try {
                    Query queryObj = QueryFactory.create(query, Syntax.syntaxARQ);

                    // Generate algebra
                    Algebra.compile(queryObj);
                    queries.put(id, queryObj);
                    len--;
                }
                catch (Exception exception){
                    exception.printStackTrace();
                    System.out.println("Error leyendo query: ");
                    continue;
                }
            }
        }
        catch (IOException e) {
            e.printStackTrace();
        }
        System.out.println("Queries readed!");
        System.out.println("Init query readed: ".concat(String.valueOf(init_row)).concat("Last readed from csv: ").concat(String.valueOf(index)));

        return  queries;
    }
    public void proccessData(String endpoint) throws IOException, InterruptedException, ExecutionException
    {
        HashMap<String, Query> queryFiles = getQueries(0, 1, outputDelimiter.toCharArray()[0]);

        ExecutorService executor = Executors.newFixedThreadPool(cores);
        ArrayList<RunSparqlQuery> tasks = new ArrayList<>();
        Iterator<Map.Entry<String, Query>> iterator = queryFiles.entrySet().iterator();
        int len = 40000;

        while (iterator.hasNext() && len > 0) {
            Map.Entry<String, Query> entry = iterator.next();
            tasks.add(new RunSparqlQuery(endpoint, entry));
            len--;
        }
        List<Future<ArrayList<String>>> futures = executor.invokeAll(tasks);
        executor.shutdown();

        try {
            executor.awaitTermination(2, TimeUnit.DAYS);
        } catch (InterruptedException e) {
            logger.info("error timeout");
        }
        System.out.println("Terminado el proceso de ejecucion");
        FileOutputStream output = new FileOutputStream(new File(outputFile));
        BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(output));
        for (Future<ArrayList<String>> future : futures)
        {
            String row = "";
            ArrayList<String> rowObj = future.get();
            if (rowObj.size() > 0 && Integer.parseInt(rowObj.get(2)) > 0) {
                row = row.concat(rowObj.get(0))  // id
                        .concat(outputDelimiter)
                        .concat(rowObj.get(1)) //query
                        .concat(outputDelimiter)
                        .concat(rowObj.get(2)) //count
                        .concat(outputDelimiter)
                        .concat(rowObj.get(3)); //time
                bw.write(row);
                bw.newLine();
            }
        }
        bw.close();
        System.out.println("Completed!!");

    }*/

    public void proccessData2(String input_delimiter_prefix, char input_delimiter) throws InterruptedException
    {

        System.out.println("Reading Queries..");
        Model model = ModelFactory.createDefaultModel();
        if (!namespaces.isEmpty()){
            model = SparqlUtils.getNamespacesFromCsv(model, namespaces, String.valueOf(input_delimiter_prefix));
        }

        Map<String, String> pref = model.getNsPrefixMap();
        Object[] keys = pref.keySet().toArray();

        boolean header = true;
        int index = 0;
        int len = len_exec_results;

        try {
            InputStreamReader csv = new InputStreamReader(new FileInputStream(inputLogsFile));
            CSVReader csvReader = new CSVReader (csv, input_delimiter);
            String[] record;
            FileOutputStream output = new FileOutputStream(new File(outputFile));
            BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(output));
            boolean iniciar = false;
            while ((record = csvReader.readNext()) != null  && len > 0) {
                index++;
                if (header || index < init_row){
                    header = false;
                    continue;
                }
                if(!iniciar){
                    System.out.println("Iniciando desde el índice: ".concat(String.valueOf(index)));
                    Thread.sleep(4000);
                    iniciar = true;
                }
                String id;
                String query = record[queryColumn];
                if(idColumn == queryColumn){
                    id = DigestUtils.md5Hex(query);
                }
                else {
                    id = record[idColumn];
                }
                try {
                    query = URLDecoder.decode(query, StandardCharsets.UTF_8.toString());
                }
                catch (Exception ex){
                    System.out.println("Query decode error, continue with query string as is.");
                }
                String prefixesStr = "";
                for (int i = 0; i < model.getNsPrefixMap().size(); i++) {

                    int a = query.indexOf(keys[i]+":");
                    if (a != -1 ) {
                        prefixesStr = prefixesStr.concat("PREFIX ").concat(String.valueOf(keys[i])).concat(": ").concat("<").concat(pref.get(String.valueOf(keys[i]))).concat("> \n");
                    }
                }
                query  = prefixesStr.concat(" " +query);
                //Pos in [0] refer to the ID of query in logs.
                try {

                    Query queryObj = QueryFactory.create(query, Syntax.syntaxARQ);

                    // Generate algebra
                    Algebra.compile(queryObj);

                    RunSparqlQuery2 obj = new RunSparqlQuery2(this.dataset, id, queryObj, outputDelimiter);
                    String rowObj = obj.getData();
                        if (rowObj.length() > 0)
                        bw.write(rowObj);
                        bw.newLine();
                        //decremento el len de resultados buscados.
                        len--;
                        System.out.println("Line: ".concat(String.valueOf(index)).concat(": ").concat(rowObj));
//                    }
                }
                catch (Exception exception){
                    System.out.println("Error leyendo query: ");
                    System.err.println(exception.toString());
                    continue;
                }

            }
            bw.close();
        }
        catch (IOException e) {
            System.err.println(e.toString());
        }
        System.out.println("Queries executed!");
        System.out.println("Init row " + init_row);
        System.out.println("Last row " + index);
    }
}
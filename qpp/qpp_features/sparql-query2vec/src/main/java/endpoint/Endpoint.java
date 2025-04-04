package endpoint;

import new_distance.GEDCalculator;
import semanticweb.RDF2GXL;
import semanticweb.sparql.KmedoidsGenerator;
import semanticweb.sparql.SparqlUtils;
import semanticweb.sparql.preprocess.*;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.concurrent.TimeUnit;

import org.apache.commons.lang3.time.StopWatch;

import lsq_extract.util.Query;
import lsq_extract.util.QueryReaderWAlgRunTime;
import lsq_extract.util.QueryReaderLegacy;
import com.org.QueryReader.LSQreader;

public class Endpoint {
    private static Map<String, String> map;

    private static void makeMap(String[] args) {
        map = new HashMap<>();
        for (String arg : args) {
            if (arg.contains("=")) {
                // works only if the key doesn't have any '='
                map.put(arg.substring(0, arg.indexOf('=')),
                        arg.substring(arg.indexOf('=') + 1));
            }
        }
    }

    public static void main(String[] args) {

        makeMap(args);

        // Getting parameters
        if (args.length == 0) {
            System.out.println("Try with some of this parameters:");
            return;
        }
        String[] params = new String[args.length - 1];
        System.arraycopy(args, 1, params, 0, args.length - 1);
        try {
            String task = args[0];
            // Define parameters from args
            boolean withHeader = map.get("--with-header") != null && !map.get("--with-header").equals("false");
            boolean withTimeFeatures = map.get("--with-timefeatures") != null
                    && !map.get("--with-timefeatures").equals("false");
            int medoids = map.get("--medoids") != null ? Integer.parseInt(map.get("--medoids")) : 25;
            // String configFile = map.get("--config-file") != null ?
            // map.get("--config-file") : "/data/config.prop";
            String urlTFPMap = map.get("--tpf-map-file") != null ? map.get("--tpf-map-file") : "";
            String prefixFile = map.get("--prefix-file") != null ? map.get("--prefix-file") : "";
            int idColumn = map.get("--idColumn") != null ? Integer.parseInt(map.get("--idColumn")) : 0;
            int cardinalityColumn = map.get("--cardinalityColumn") != null
                    ? Integer.parseInt(map.get("--cardinalityColumn"))
                    : 8;
            int queryColumn = map.get("--queryColumn") != null ? Integer.parseInt(map.get("--queryColumn")) : 1;
            int cores = map.get("--cores") != null ? Integer.parseInt(map.get("--cores")) : 12;
            int length = map.get("--length") != null ? Integer.parseInt(map.get("--length")) : 0;
            int elemtByCore = map.get("--elements-by-core") != null ? Integer.parseInt(map.get("--elements-by-core"))
                    : 1000;
            String output_delimiter = map.get("--output-delimiter") != null ? map.get("--output-delimiter") : ",";
            // Delimitador de elementos dentro de una columna, la coma no es buena debido a
            // que existen uris con coma.
            String output_element_delimiter = map.get("--output-element-delimiter") != null
                    ? map.get("--output-element-delimiter")
                    : "ᶷ";
            String input_delimiter = map.get("--input-delimiter") != null ? map.get("--input-delimiter") : "~";
            int execTimeColumn = map.get("--execTimeColumn") != null ? Integer.parseInt(map.get("--execTimeColumn"))
                    : 7;
            String query = map.get("--query") != null ? map.get("--query") : "";

            int indexStart = map.get("--indexStart") != null ? Integer.parseInt(map.get("--indexStart")) : 0;
            int indexEnd = map.get("--indexEnd") != null ? Integer.parseInt(map.get("--indexEnd")) : 1000000000;
            // Environment variable hack instead of configuration file
            String configFile = System.getenv("config");
            String inputQueryFile = map.get("--input-queryfile") != null ? map.get("--input-queryfile") : null; //for optimized dist computation
            String outputQueryFile = map.get("--output-queryfile") != null ? map.get("--output-queryfile") : null; //for optimized dist computation


	    boolean isWDBench = map.get("--isWDBench") != null ? Boolean.parseBoolean(map.get("--isWDBench")): false;

            switch (task) {
                case "ged-opt": {
                    if(inputQueryFile == null || outputQueryFile == null){
                        throw new Exception("--input-queryfile or --output-queryfile were not specified");
                    }
                    GEDCalculator calculator = new GEDCalculator();
                    List<Map<String,String>> map = calculator.readJson(inputQueryFile);
                    List<Map<String,String>> distMap = calculator.makeDistJson(map);
                    calculator.writeDist(distMap, outputQueryFile);
                    break;
                }
                case "kmedoids": {
                    System.out.println("Entering to kmedoids class");
                    KmedoidsGenerator kmedoidsGenerator = new KmedoidsGenerator();

                    kmedoidsGenerator.proccessQueries(params, medoids, input_delimiter,
                            output_delimiter.toCharArray()[0], idColumn, execTimeColumn, withHeader);
                    break;
                }
                case "edit-distance": {
                    SparqlUtils sparqlUtils = new SparqlUtils();
                    String input, output;
                    try {
                        input = params[0];
                        output = params[1];
                    } catch (Exception ex) {
                        System.out.println("args[0] : Input csv \n args[1] : Output path \n");
                        return;
                    }
                    // int coress = Runtime.getRuntime().availableProcessors();
                    sparqlUtils.calculateEditDistance(input, output, prefixFile.equals("") ? null : prefixFile, cores,
                            input_delimiter.toCharArray()[0], output_delimiter.toCharArray()[0], idColumn, queryColumn,
                            execTimeColumn, elemtByCore);
                    break;
                }
                case "lsq-edit-distance": {
                    // ReinforcementLearningExtractor rpfv = new ReinforcementLearningExtractor();
                    QueryReaderWAlgRunTime.newLSQdata = true;
                    lsq_extract.util.QueryReaderWAlgRunTime reader = new lsq_extract.util.QueryReaderWAlgRunTime(
                            params[0]);
                    try {
                        reader.readFile();
                    } catch (IOException e) {
                        System.out.println("Something went wrong with the file path!");
                    }
                    reader.close();
                    StopWatch s = new StopWatch();
                    s.start();
                    SparqlUtils sparqlUtils = new SparqlUtils();
                    sparqlUtils.calculateEditDistance(reader.qs, params[1], cores, '\t', elemtByCore, indexStart,
                            indexEnd);
                    s.stop();
                    System.out.println("Calculating the GED took " + s.getTime() + " MS");
                    break;
                }
                // 1 center queries, 2 train log, 3 val log, 3 test log
                case "train_distance": {
                    SparqlUtils sparqlUtils = new SparqlUtils();
                    ArrayList<String> centerQueries = getQueryFromTextFile(params[0]);
                    ArrayList<Query> trainQueries = getQueryFromFile(params[1]);
                    ArrayList<Query> valQueries = getQueryFromFile(params[2]);
                    ArrayList<Query> testQueries = getQueryFromFile(params[3]);
                    sparqlUtils.calculateEditDistanceTrainQueries(centerQueries, trainQueries,
                            params[1].substring(0, params[1].length() - 4) + "_graph.txt");
                    sparqlUtils.calculateEditDistanceTrainQueries(centerQueries, valQueries,
                            params[2].substring(0, params[2].length() - 4) + "_graph.txt");
                    sparqlUtils.calculateEditDistanceTrainQueries(centerQueries, testQueries,
                            params[3].substring(0, params[3].length() - 4) + "_graph.txt");

                    break;
                }
                case "graph-patterns": {
                    // Compute edit distances only with a list of medoids ids file and output the
                    // graph pattern.
                    SparqlUtils sparqlUtils = new SparqlUtils();
                    String input, output, modeidsFile;
                    try {
                        input = params[0];
                        output = params[1];
                        modeidsFile = params[2];

                    } catch (Exception ex) {
                        System.out.println("args[0] : Input csv \n args[1] : Output path \n");
                        return;
                    }
                    sparqlUtils.getGraphPatterns(input, output, modeidsFile, prefixFile.equals("") ? null : prefixFile,
                            input_delimiter.toCharArray()[0], output_delimiter.toCharArray()[0], idColumn, queryColumn,
                            execTimeColumn);
                    break;
                }
                case "algebra-features": {
                    String inputFile = params[0];
                    String outputFile = params[1];
		    QueryReaderLegacy.isWDbench = isWDBench;
                    TDBExecutionAndFeature.produceALgebraFeatures(inputFile, outputFile, prefixFile,
                            input_delimiter, output_delimiter, idColumn, queryColumn, execTimeColumn);
                    break;
                }
                case "rlearning": {
                    ReinforcementLearningExtractor rpfv = new ReinforcementLearningExtractor();
                    rpfv.getArrayFeaturesVector(params[0], params[1], prefixFile, queryColumn, idColumn, execTimeColumn,
                            length, "ᶶ", input_delimiter, output_element_delimiter);
                    break;
                }
                // input file has to be tab seperated!
                case "extra": { // renamed from rlearning to extra
                    ReinforcementLearningExtractor rpfv = new ReinforcementLearningExtractor();
                    QueryReaderWAlgRunTime.newLSQdata = true;
                    QueryReaderWAlgRunTime.parseQueryString = false;
                    QueryReaderWAlgRunTime.extra=true;
                    ArrayList<Query> qs= new ArrayList<>();
                    /*lsq_extract.util.QueryReaderWAlgRunTime reader = new lsq_extract.util.QueryReaderWAlgRunTime(
                            params[0]);*/
                    lsq_extract.util.QueryReaderWAlgRunTime reader = new lsq_extract.util.QueryReaderWAlgRunTime(
                            params[0]);
                    /*com.org.QueryReader.LSQreader reader = new LSQreader(
                            params[0]);*/

                    try {
                        qs = reader.readFile();
                    } catch (IOException e) {
                        System.out.println("Something went wrong with the file path!");
                    }
                    reader.close();
                    rpfv.getArrayFeaturesVector(qs, params[1], "??44??");
                    break;
                }
                case "algebra-feature-query": {
                    StringBuilder sb = new StringBuilder();
                    String[] header = new String[] {
                            "triple", "bgp", "join", "leftjoin", "union", "filter", "graph", "extend", "minus", "path*",
                            "pathN*", "path+", "pathN+", "path?", "notoneof", "tolist", "order", "project", "distinct",
                            "reduced",
                            "multi", "top", "group", "assign", "sequence", "slice", "treesize" };
                    AlgebraFeatureExtractor fe = new AlgebraFeatureExtractor(header);
                    double[] features = fe.extractFeatures(query);
                    sb.append("{");
                    int i = 0;
                    while (i < header.length) {
                        sb.append("\"");
                        sb.append(header[i]);
                        sb.append("\":");
                        sb.append(features[i]);
                        sb.append(",");
                        i++;
                    }
                    sb.append("}");
                    System.out.println(sb.toString());
                    break;
                }
                case "algebra-extract":
                    String path = args[1];
                    ArrayList<String> queries = new ArrayList<>();
                    ArrayList<String> ids = new ArrayList<>();
                    String[] header = new String[] {
                            "triple", "bgp", "join", "leftjoin", "union", "filter", "graph", "extend", "minus", "path*",
                            "pathN*", "path+", "pathN+", "path?", "notoneof", "tolist", "order", "project", "distinct",
                            "reduced",
                            "multi", "top", "group", "assign", "sequence", "slice", "treesize" };
                    try {
                        Scanner scanner = new Scanner(new File(path));
                        scanner.nextLine();
                        while (scanner.hasNextLine()) {
                            String line = scanner.nextLine();
                            String[] splits = line.split(",", 2);
                            ids.add(splits[0]);
                            queries.add(splits[1]);
                        }
                    } catch (FileNotFoundException e) {
                        throw new RuntimeException(e);
                    }
                    try {
                        FileWriter writer = new FileWriter(new File(args[2]));
                        writer.write("id,");
                        otheralgebra.AlgebraFeatureExtractor z = new otheralgebra.AlgebraFeatureExtractor();
                        List<Map.Entry<String, Integer>> ll = new ArrayList<Map.Entry<String, Integer>>(
                                z.getFeatureIndex().entrySet());
                        // fe.featureIndex.entrySet();

                        Collections.sort(ll, new Comparator<Map.Entry<String, Integer>>() {

                            @Override
                            public int compare(Map.Entry<String, Integer> o1,
                                    Map.Entry<String, Integer> o2) {
                                // TODO Auto-generated method stub
                                return o1.getValue() - o2.getValue();
                            }
                        });
                        StringBuilder temp = new StringBuilder();
                        for (Map.Entry<String, Integer> e : ll) {
                            temp.append(e.getKey()).append(",");
                        }
                        temp.deleteCharAt(temp.length() - 1);
                        temp.append("\n");
                        writer.write(temp.toString());

                        double[] features;
                        for (int i = 0; i < queries.size(); i++) {
                            StringBuilder feat = new StringBuilder();
                            feat.append(ids.get(i));
                            feat.append(",");
                            features = z.extractFeatures(queries.get(i));

                            for (double f : features) {
                                feat.append(f);
                                feat.append(",");
                            }
                            feat.deleteCharAt(feat.length() - 1);
                            feat.append("\n");
                            writer.write(feat.toString());
                            writer.flush();
                        }
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }

                    break;
                default: {
                    System.out.println(
                            "The task not found. Pleas use one of them: 'kmedoids, edit-distance, algebra-features, predicate-features'");
                }
            }
        } catch (Exception ex) {
            ex.printStackTrace();
            System.out.println("You need to specify a task as first parameter");
        }
    }

    public static ArrayList<Query> getQueryFromFile(String path) {
        QueryReaderWAlgRunTime.newLSQdata = true;
        QueryReaderWAlgRunTime.parseQueryString = false;
        lsq_extract.util.QueryReaderWAlgRunTime reader = new lsq_extract.util.QueryReaderWAlgRunTime(path);
        try {
            reader.readFile();
        } catch (IOException e) {
            System.out.println("Something went wrong with the file path!");
        }
        reader.close();
        return reader.getQs();
    }

    public static ArrayList<String> getQueryFromTextFile(String path) {
        ArrayList<String> qs = new ArrayList<>();
        try {
            File myObj = new File(path);
            Scanner myReader = new Scanner(myObj);
            while (myReader.hasNextLine()) {
                String data = myReader.nextLine();
                if (data.isBlank()) {
                    continue;
                }
                qs.add(data);
            }
            myReader.close();
        } catch (FileNotFoundException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        }
        return qs;
    }
}

package semanticweb.sparql.preprocess;

import org.apache.jena.query.*;
import org.apache.jena.rdf.model.Model;
import org.apache.jena.rdf.model.ModelFactory;
import org.apache.jena.rdf.model.RDFNode;
import org.apache.jena.rdf.model.impl.LiteralImpl;
import org.apache.jena.sparql.algebra.Algebra;
import org.apache.jena.sparql.algebra.Op;
import semanticweb.sparql.SparqlUtils;
import semanticweb.sparql.utils.PredHistogram;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class ReinforcementLearningExtractor extends FeaturesExtractorBase {
    public static Model model;

    /**
     * Retrieve vectors For use the tpf of queries as sequences for recurrent approach.
     *
     * @param urlQueries       Url for queries file.
     * @param output           Url for output file.
     * @param namespaces       Url for Sparql prefixes file.
     * @param queryColumn      query column position.
     * @param idColumn         IdColumn position.
     * @param execTimeColumn   execution time column position.
     * @param length           Length of queries to get from the csv queries file.
     * @param output_delimiter Delimiter for csv file column.
     * @param input_delimiter  Delimiter for csv file column.
     * @return ArrayList with Map of queries data generated, see @{@link QueryFeatureExtractor}
     */
    public ArrayList<Map<String, Object>> getArrayFeaturesVector(String urlQueries,
                                                                 String output,
                                                                 String namespaces,
                                                                 int queryColumn,
                                                                 int idColumn,
                                                                 int execTimeColumn,
                                                                 int length,
                                                                 String output_delimiter,
                                                                 String input_delimiter,
                                                                 String output_element_delimiter) {

        ArrayList<Map<String, Object>> vectors = new ArrayList<>();

        ArrayList<String[]> featInQueryList = SparqlUtils.getQueries(urlQueries, namespaces , new ArrayList<>(), idColumn, queryColumn, execTimeColumn, input_delimiter.toCharArray()[0],"./unprocessed.txt");

        //we use the size of array intead of -1(csv header) because we use extra column called others.
        HashMap<String, PredHistogram> predicatesToHist = new HashMap<>();
        int indexPredJoins = 0;
        for (String[] queryArr : featInQueryList) {
            try {
                RLQueryFeatureExtractor qfe = new RLQueryFeatureExtractor(queryArr);
                Map<String, Object> queryVecData = qfe.getProcessedData();
                vectors.add(queryVecData);
                HashMap<String, PredHistogram> queriesToHist = ((HashMap) queryVecData.get("queriesToHist"));
                predicatesToHist = addQueryHistToCollection(queriesToHist, predicatesToHist);
            } catch (Exception ex) {
                ex.printStackTrace();
            }
        }
        ArrayList<String> vectorHeader = new ArrayList<>();
        vectorHeader.add("id");
        vectorHeader.add("joins");
        vectorHeader.add("predicates");
        vectorHeader.add("execTime");
        produceCsvArrayVectors(vectorHeader, vectors, output, output_delimiter, output_element_delimiter);
        produceHistFile(predicatesToHist, output, output_delimiter);
        return vectors;
    }
    public ArrayList<Map<String, Object>> getArrayFeaturesVector(ArrayList<lsq_extract.util.Query> queries,
                                                                 String outputPath,
                                                                 String output_element_delimiter) {

        ArrayList<Map<String, Object>> vectors = new ArrayList<>();

        HashMap<String, PredHistogram> predicatesToHist = new HashMap<>();
        int indexPredJoins = 0;
        for (lsq_extract.util.Query q : queries) {
            try {
                RLQueryFeatureExtractor qfe = new RLQueryFeatureExtractor(new String[] {q.id,q.text,Double.toString(q.duration)});
                Map<String, Object> queryVecData = qfe.getProcessedData();
                vectors.add(queryVecData);
                HashMap<String, PredHistogram> queriesToHist = ((HashMap) queryVecData.get("queriesToHist"));
                predicatesToHist = addQueryHistToCollection(queriesToHist, predicatesToHist);
            } catch (Exception ex) {
                ex.printStackTrace();
            }
        }
        ArrayList<String> vectorHeader = new ArrayList<>();
        vectorHeader.add("id");
        vectorHeader.add("joins");
        vectorHeader.add("predicates");
        vectorHeader.add("execTime");
        produceCsvArrayVectors(vectorHeader, vectors, outputPath, "\t", output_element_delimiter);
        produceHistFile(predicatesToHist, outputPath, "\t");
        return vectors;
    }

    /**
     * Produce un listado de Predicado,type,on,query valores respectivos a un sampling para ser luego ejecutado.
     * @param predicatesToHist HashMap<String, PredHistogram>  hash con la uri del predicado como llave y su Histograma data
     * @param output String Dirección de salida
     * @param output_delimiter String Caracter de delimitación de valores para el csv a generar.
     */
    public static void produceHistFile(HashMap<String, PredHistogram> predicatesToHist, String output, String output_delimiter) {
        BufferedWriter br;
        StringBuilder sb = new StringBuilder();
        // Using for-each loop
        for (PredHistogram histogram : predicatesToHist.values()) {
            sb.append(histogram.toSamplingQueryFileString(output_delimiter));
            sb.append("\n");
        }
        try {
            br = new BufferedWriter(new FileWriter(output.concat(".sampling_file").concat(".txt")));
            br.write(sb.toString());
            br.close();
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    public HashMap<String, PredHistogram> addQueryHistToCollection(HashMap<String, PredHistogram> queriesToHist, HashMap<String, PredHistogram> globalHistCollection) {
        String[] histCurr = (String[]) queriesToHist.keySet().toArray(new String[0]);
        for (String s : histCurr) {
            PredHistogram predHistCurr = queriesToHist.get(s);
            if (globalHistCollection.containsKey(s)) {
                PredHistogram globalHist = globalHistCollection.get(s);

                if (predHistCurr.getOnSubQuery() != null) {
                    globalHist.setQuery("sub", predHistCurr.getOnSubType(), predHistCurr.getOnSubQuery());
                }
                if (predHistCurr.getOnObjQuery() != null) {
                    globalHist.setQuery("obj", predHistCurr.getOnObjType(), predHistCurr.getOnObjQuery());
                }
                globalHistCollection.put(s, globalHist);
            } else {
                globalHistCollection.put(s, predHistCurr);
            }
        }
        return globalHistCollection;
    }
    public static boolean isStringInt(String s)
    {
        try
        {
            Integer.parseInt(s);
            return true;
        } catch (NumberFormatException ex)
        {
            return false;
        }
    }

    /**
     * Create a csv with array data passed as parameters.
     *
     * @param headers          {@link ArrayList} With header for output file to generate
     * @param list             Queries data list.
     * @param filepath         Output file path.
     * @param output_delimiter Delimiter for csv data columns.
     */
    public static void produceCsvArrayVectors(ArrayList<String> headers, ArrayList<Map<String, Object>> list, String filepath, String output_delimiter, String output_element_delimiter) {
        BufferedWriter br;
        BufferedWriter brpred_uri;
        try {
            StringBuilder sb = new StringBuilder();
            StringBuilder sbbrpred_uri = new StringBuilder();
            // Append strings from array
            for (String element : headers) {
                sb.append(element);
                sb.append(output_delimiter);
            }
            sb.append("\n");
            int count = 0;
            for (Map<String, Object> queryData : list) {
                for (String header : headers) {
                    switch (header) {
                        case "tpfs": {
                            ArrayList<HashMap<String, String>> qTables = (ArrayList<HashMap<String, String>>) queryData.get("tpfs");
                            if (qTables.size() > 0) {
                                for (HashMap<String, String> element : qTables) {
                                    sb.append(element.get("structure"));
                                    sb.append(",");
                                    sb.append(element.get("predicate"));
                                    sb.append(",");
                                }
                            } else {
                                sb.append("EMPTY_VALUE");
                            }
                            break;
                        }
                        case "joins": {
                            ArrayList<ArrayList> qTables = (ArrayList<ArrayList>) queryData.get("queryJoinsVec");
                            if (qTables.size() > 0) {
                                try {
                                    for (ArrayList<String> elementColOpCard : qTables) {
                                        sb.append(elementColOpCard.get(0));
                                        sb.append(output_element_delimiter);
                                        sb.append(elementColOpCard.get(1));
                                        sb.append(output_element_delimiter);
                                    }
                                } catch (Exception ex) {
                                    ex.printStackTrace();
                                    System.out.println(qTables);
                                }
                            } else {
                                //Todo ver que hacer en caso de que no existan joins
                                // sb.append("EMPTY_VALUE");
                            }
                            break;
                        }
                        case "predicates": {
                            ArrayList<HashMap<String, String>> queryPredicates = (ArrayList<HashMap<String, String>>) queryData.get("queryPredicates");
                            if (queryPredicates.size() > 0) {
                                try {
                                    for (HashMap<String, String> elementColOpCard : queryPredicates) {

                                            sb.append(elementColOpCard.get("col"));
                                            sb.append(output_element_delimiter);
                                            sb.append(elementColOpCard.get("operator"));
                                            sb.append(output_element_delimiter);
                                            sb.append(elementColOpCard.get("on"));
                                            sb.append(output_element_delimiter);
                                            sb.append(String.valueOf( elementColOpCard.get("value")));
                                            sb.append(output_element_delimiter);

                                    }
                                } catch (Exception ex) {
                                    ex.printStackTrace();
                                    System.out.println(queryPredicates);
                                }
                            } else {
                                //Todo ver que hacer en caso de que no existan joins
                                // sb.append("EMPTY_VALUE");
                            }

                            ArrayList<HashMap<String, String>> queryPredicatesUri = (ArrayList<HashMap<String, String>>) queryData.get("queryPredicatesUris");
                            if (queryPredicatesUri.size() > 0) {
                                try {
                                    for (HashMap<String, String> elementColOpCard : queryPredicatesUri) {
                                        sb.append(elementColOpCard.get("col"));
                                        sb.append(output_element_delimiter);
                                        sb.append(elementColOpCard.get("operator"));
                                        sb.append(output_element_delimiter);
                                        sb.append(elementColOpCard.get("on"));
                                        sb.append(output_element_delimiter);
                                        sb.append(String.valueOf( elementColOpCard.get("value")));
                                        sb.append(output_element_delimiter);
                                    }
                                } catch (Exception ex) {
                                    ex.printStackTrace();
                                    System.out.println(queryPredicates);
                                }
                            } else {
                                //Todo ver que hacer en caso de que no existan joins
                                // sb.append("EMPTY_VALUE");
                            }
                            break;
                        }
                        default:{
                            sb.append(queryData.get(header));
                        }
                    }
                    // Add separator
                    sb.append(output_delimiter);
                }
                sb.append("\n");
                count++;
                if (count % 1000 == 0) {
                    brpred_uri = new BufferedWriter(new FileWriter(filepath.concat(".preduri.").concat(String.valueOf(count)).concat(".txt")));
                    brpred_uri.write(sbbrpred_uri.toString());
                    brpred_uri.close();
                }
            }
            br = new BufferedWriter(new FileWriter(filepath));
            brpred_uri = new BufferedWriter(new FileWriter(filepath.concat(".preduri.txt")));
            br.write(sb.toString());
            brpred_uri.write(sbbrpred_uri.toString());
            br.close();
            brpred_uri.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void executeSamplingHist(String url, String output, String output_delimiter, String input_delimiter, String output_element_delimiter, String prefixFile) {
        ArrayList<String[]> arrays = SparqlUtils.getArrayFromCsvFile(url, input_delimiter);
        HashMap<String, PredHistogram> predToHistEvaluated = new HashMap<>();
        for (String[] array : arrays) {
            String key = array[0];
            String targetType = array[1];
            String on = array[2];
            String queryString = array[3];

            Query query = QueryFactory.create(String.valueOf(queryString));

            // Generate algebra
            Op op = Algebra.compile(query);
            op = Algebra.optimize(op);
            QueryExecution exec = QueryExecutionFactory.sparqlService("http://dbpedia.org/sparql", query);

            ResultSet results = exec.execSelect();
            HashMap<String, Double> values = new HashMap<>();
            if (targetType.equals("numeric") && results.hasNext()) {
                QuerySolution result = results.next();
                RDFNode min = result.get("min");
                RDFNode max = result.get("max");
                RDFNode distinct = result.get("distinct");
                try {
                    values.put("min", ((LiteralImpl) min).getDouble());
                    values.put("max", ((LiteralImpl) max).getDouble());
                    values.put("distinct", ((LiteralImpl) distinct).getDouble());
                } catch (Exception ex) {
                    System.out.println("Calculando min max");
                    ex.printStackTrace();
                }
            } else {
                while (results.hasNext()) {
                    QuerySolution result = results.next();
                    RDFNode val = result.get("sub");
                    if (val == null) {
                        val = result.get("obj");
                    }
                    String valString = val.toString().split("\\^\\^")[0];
                    if (values.containsKey(valString)) {
                        values.put(valString, values.get(valString) + 1);
                    } else {
                        values.put(valString, 1.0);
                    }
                }
            }
            if (predToHistEvaluated.containsKey(key)) {
                PredHistogram predHistogram = predToHistEvaluated.get(key);
                predHistogram.setValuesHist(on, targetType, values);
                predToHistEvaluated.put(key, predHistogram);
            } else {
                PredHistogram predHistogram = new PredHistogram(targetType, on, values, key);
                predToHistEvaluated.put(key, predHistogram);
            }
        }
        produceHisValuesFile(predToHistEvaluated, output, output_delimiter, output_element_delimiter);
    }

    public static void produceHisValuesFile(HashMap<String, PredHistogram> predToHistEvaluated, String output, String output_delimiter, String output_element_delimiter) {
        BufferedWriter br;
        StringBuilder sb = new StringBuilder();
        // Using for-each loop
        for (String key : predToHistEvaluated.keySet().toArray(new String[0])) {

            // Add some bonus marks
            // to all the students and print it
            PredHistogram predHistogram = predToHistEvaluated.get(key);
            sb.append(predHistogram.printHistDataFileString(output_delimiter, output_element_delimiter));
            sb.append("\n");
        }
        try {
            br = new BufferedWriter(new FileWriter(output.concat(".valued").concat(".txt")));
            br.write(sb.toString());
            br.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
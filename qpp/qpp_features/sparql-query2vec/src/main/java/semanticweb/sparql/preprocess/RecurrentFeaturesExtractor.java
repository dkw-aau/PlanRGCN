package semanticweb.sparql.preprocess;

import org.apache.commons.codec.digest.DigestUtils;
import semanticweb.sparql.SparqlUtils;
import semanticweb.sparql.utils.DBPediaUtils;

import java.io.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class RecurrentFeaturesExtractor  extends DeepSetFeatureExtractor{


    /**
     * Retrieve vectors for For use the tpf of queries as sequences for recurrent approach.
     *
     * @param urlQueries        Url for queries file.
     * @param output            Url for output file.
     * @param namespaces        Url for Sparql prefixes file.
     * @param queryColumn       query column position.
     * @param idColumn          IdColumn position.
     * @param execTimeColumn  execution time column position.
     * @param length            Length of queries to get from the csv queries file.
     * @param output_delimiter  Delimiter for csv file column.
     * @param input_delimiter   Delimiter for csv file column.
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
                                                                        String input_delimiter) {

        if(!namespaces.equals("false")){
            model = SparqlUtils.getNamespacesDBPed(namespaces);
        }
        ArrayList<Map<String, Object>> vectors = new ArrayList<>();

        ArrayList<ArrayList<String>> featInQueryList = this.getArrayQueriesMetaFromCsv(urlQueries, true, input_delimiter, queryColumn, idColumn, execTimeColumn, length);
        //we use the size of array intead of -1(csv header) because we use extra column called others.
        for (ArrayList<String> queryArr : featInQueryList) {
            try {
                RecurrentQueryFeatureExtractor qfe = new RecurrentQueryFeatureExtractor(queryArr);
                Map<String, Object> queryVecData = qfe.getProcessedData();
                vectors.add(queryVecData);
            } catch (Exception ex) {
                ex.printStackTrace();
            }

        }
        ArrayList<String> vectorHeader = new ArrayList<>();
        vectorHeader.add("id");
        vectorHeader.add("tpfs");
        vectorHeader.add("execTime");
        produceCsvArrayVectors(vectorHeader, vectors, output, output_delimiter);
        return vectors;
    }


    /**
     * Create a csv with array data passed as parameters.
     *
     * @param headers          {@link ArrayList} With header for output file to generate
     * @param list             Queries data list.
     * @param filepath         Output file path.
     * @param output_delimiter Delimiter for csv data columns.
     */
    public static void produceCsvArrayVectors(ArrayList<String> headers, ArrayList<Map<String, Object>> list, String filepath, String output_delimiter) {
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
            HashMap<String, String> namespases = SparqlUtils.getNamespacesStr("/home/daniel/Documentos/Web_Semantica/Work/Sparql2vec/prefixes.txt");
            HashMap<String, Integer> tpfCardinalities = new HashMap<>();

            String ENDPOINT = "https://dbpedia.org/sparql";
            int count = 0;
            for (Map<String, Object> queryData : list) {
                for (String header : headers) {
                    switch (header) {
                        case "tpfs": {
                            ArrayList<HashMap<String,String>> qTables = (ArrayList<HashMap<String,String>>) queryData.get("tpfs");
                            if (qTables.size() > 0) {
                                for (HashMap<String,String> element : qTables) {
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
//                        case "joins": {
//                            ArrayList<String> qTables = (ArrayList<String>) queryData.get("queryJoins");
//                            if (qTables.size() > 0) {
//                                for (String element : qTables) {
//                                    sb.append(element);
//                                    sb.append(",");
//                                }
//                            } else {
//                                sb.append("EMPTY_VALUE");
//                            }
//                            break;
//                        }
//                        case "joins_v1": {
//                            ArrayList<ArrayList> qTables = (ArrayList<ArrayList>) queryData.get("queryJoinsV1Vec");
//                            if (qTables.size() > 0) {
//                                try {
//                                    for (ArrayList<String> elementColOpCard : qTables) {
//                                        sb.append(elementColOpCard.get(0));
//                                        sb.append(",");
//                                        sb.append(elementColOpCard.get(1));
//                                        sb.append(",");
//                                        sb.append(elementColOpCard.get(2));
//                                        sb.append(",");
//                                    }
//                                } catch (Exception ex) {
//                                    ex.printStackTrace();
//                                    System.out.println(qTables);
//                                }
//                            } else {
//                                sb.append("EMPTY_VALUE");
//                            }
//                            break;
//                        }
//                        case "predicates_v2int": {
//                            ArrayList<HashMap<String, Object>> qPredicates = (ArrayList<HashMap<String, Object>>) queryData.get("queryPredicates");
//                            if (qPredicates.size() > 0) {
//                                for (HashMap<String, Object> element : qPredicates) {
//                                    sb.append(element.get("col"));
//                                    sb.append(",");
//                                    sb.append(element.get("operator"));
//                                    sb.append(",");
//                                    sb.append(element.get("value"));
//                                    sb.append(",");
//                                }
//                            } else {
//                                sb.append("EMPTY_VALUE");
//                            }
//                            break;
//                        }
//                        case "predicates_v2uri": {
//                            ArrayList<HashMap<String, Object>> qPredicates = (ArrayList<HashMap<String, Object>>) queryData.get("queryPredicatesUris");
//                            if (qPredicates.size() > 0) {
//                                for (HashMap<String, Object> element : qPredicates) {
//                                    sb.append(element.get("col"));
//                                    sb.append(",");
//                                    sb.append(element.get("operator"));
//                                    sb.append(",");
//                                    sb.append(element.get("object"));
//                                    sb.append(",");
//                                }
//                            } else {
//                                sb.append("EMPTY_VALUE");
//                            }
//                            break;
//                        }
//                        case "pred_v2uri_cardinality": {
//                            ArrayList<HashMap<String, Object>> qPredicates = (ArrayList<HashMap<String, Object>>) queryData.get("queryPredicatesUris");
//                            if (qPredicates.size() > 0) {
//                                for (HashMap<String, Object> element : qPredicates) {
//
//                                    sb.append(element.get("col"));
//                                    sb.append(",");
//                                    sb.append(element.get("operator"));
//                                    sb.append(",");
//                                    sb.append(element.get("object"));
//                                    sb.append(",");
//                                    String hash = String.valueOf(element.get("sampling_query_id"));
//                                    hash = Hashing.md5().hashString(hash).toString();
//                                    if (tpfCardinalities.get(hash) != null) {
//                                        int card = tpfCardinalities.get(hash);
//                                        sb.append(card);
//                                        sb.append(",");
//                                    } else {
//                                        String query = String.valueOf(element.get("sampling_query"));
//                                        int result = 0;
//                                        try {
//                                            result = DBPediaUtils.execQuery(query, ENDPOINT, namespases);
//                                        } catch (Exception ex) {
//                                            ex.printStackTrace();
//                                            System.out.println("Query failed: ".concat(hash));
//                                        }
//
//                                        sb.append(result);
//                                        sb.append(",");
//                                        tpfCardinalities.put(hash, result);
//                                        sbbrpred_uri.append(hash.concat(" , ".concat(String.valueOf(result))));
//                                        sbbrpred_uri.append("\n");
//                                    }
//                                }
//                            } else {
//                                sb.append("EMPTY_VALUE");
//                            }
//                            break;
//                        }
                        case "id":
                        case "cardinality":
                        case "execTime": {
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
}

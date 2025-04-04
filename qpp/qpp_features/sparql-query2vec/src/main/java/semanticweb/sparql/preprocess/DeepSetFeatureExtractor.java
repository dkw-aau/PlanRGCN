package semanticweb.sparql.preprocess;

import org.apache.commons.codec.digest.DigestUtils;
import org.apache.jena.rdf.model.Model;
import semanticweb.RecursiveDeepSetFeaturizeAction;
import semanticweb.sparql.SparqlUtils;
import semanticweb.sparql.utils.DBPediaUtils;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ForkJoinPool;


public class DeepSetFeatureExtractor extends  FeaturesExtractorBase{
    public static Model model;
    public static String prefixes = "";

    public static ArrayList<String> default_sets = new ArrayList<>(Arrays.asList("tables", "joins", "joins_v1", "predicates_v2int", "predicates_v2uri", "pred_v2uri_cardinality"));
    public ArrayList<String> tablesOrder;
    public ArrayList<String> joinsOrder;
    public ArrayList<String> predicatesOrder;
    public ArrayList<String> predicatesUrisOrder;
    public HashMap<String, Integer> predUrisCardinality;

    public DeepSetFeatureExtractor() {
        this.tablesOrder = new ArrayList<>();
        this.joinsOrder = new ArrayList<>();
        this.predicatesOrder = new ArrayList<>();
        this.predicatesUrisOrder = new ArrayList<>();
        this.predUrisCardinality = new HashMap<>();
//        default_sets.addAll(Arrays.asList("tables","joins","predicates_v2int", "predicates_v2uri"));
    }

    /**
     * Retrieve vectors for DeepSet architecture with fixed column indexes
     *
     * @param urlQueries       Url for queries file.
     * @param output           Url for output file.
     * @param namespaces       Url for Sparql prefixes file.
     * @param length           Length of queries to get from the csv queries file.
     * @param output_delimiter Delimiter for csv file column.
     * @param input_delimiter  Delimiter for csv file column.
     * @return ArrayList with Map of queries data generated, see @{@link QueryFeatureExtractor}
     */
    public ArrayList<Map<String, Object>> getArrayFeaturesVector(String urlQueries, String output, String sets, String namespaces, int length, String output_delimiter, String input_delimiter, String urlTFPMap) {

        return getArrayFeaturesVector(urlQueries, output, sets, namespaces, 1, 0, 8, length, output_delimiter, input_delimiter, urlTFPMap);
    }

    /**
     * Retrieve vectors for DeepSet architecture.
     *
     * @param urlQueries        Url for queries file.
     * @param output            Url for output file.
     * @param namespaces        Url for Sparql prefixes file.
     * @param queryColumn       query column position.
     * @param idColumn          IdColumn position.
     * @param cardinalityColumn cardinality column position.
     * @param length            Length of queries to get from the csv queries file.
     * @param output_delimiter  Delimiter for csv file column.
     * @param input_delimiter   Delimiter for csv file column.
     * @return ArrayList with Map of queries data generated, see @{@link QueryFeatureExtractor}
     */
    public ArrayList<Map<String, Object>> getArrayFeaturesVector(String urlQueries, String output, String sets,
                                                                        String namespaces, int queryColumn, int idColumn,
                                                                        int cardinalityColumn, int length,
                                                                        String output_delimiter,
                                                                        String input_delimiter,
                                                                        String urlTFPMap) {

        if(!namespaces.equals("false")){
            model = SparqlUtils.getNamespacesDBPed(namespaces);
        }
        ArrayList<Map<String, Object>> vectors = new ArrayList<>();

        ArrayList<String> featuresArray = new ArrayList<>();
        String[] arraySets = sets.split(",");
        // Add trusted features set to list.
        if (arraySets.length > 1) {
            for (String arraySet : arraySets) {
                if (default_sets.contains(arraySet)) {
                    featuresArray.add(arraySet);
                }
            }
        }
        String extract_tpf_dir = "/mnt/46d157e4-f3b2-4033-90d8-ed2b2b56139e/wadiv_data/tpf_queries";
        HashMap<String,String> tpf_card_queries = new HashMap<>();
        ArrayList<ArrayList<String>> featInQueryList = this.getArrayQueriesMetaFromCsv(urlQueries, true, input_delimiter, queryColumn, idColumn, cardinalityColumn, length);
        //we use the size of array intead of -1(csv header) because we use extra column called others.
        for (ArrayList<String> queryArr : featInQueryList) {
            try {
                QueryFeatureExtractor qfe = new QueryFeatureExtractor(queryArr);
                Map<String, Object> queryVecData = qfe.getProcessedData();
                vectors.add(queryVecData);
                ArrayList qpu = (ArrayList) queryVecData.get("queryPredicatesUris");
//                if(true){
//                    for (int i = 0; i < qpu.size(); i++) {
//                        HashMap<String,String> temp = (HashMap<String, String>) qpu.get(i);
//                        tpf_card_queries.put(temp.get("sampling_query_id"),temp.get("sampling_query"));
//                        String fileTFPName = extract_tpf_dir.concat("/").concat(Hashing.md5().hashString(temp.get("sampling_query_id")).toString());
//                        BufferedWriter br = new BufferedWriter(new FileWriter(fileTFPName));
//                        br.write(temp.get("sampling_query"));
//                        br.close();
//                    }
//                }

            } catch (Exception ex) {
                ex.printStackTrace();
            }

        }
        ArrayList<String> vectorHeader = new ArrayList<>();
        vectorHeader.add("id");
        vectorHeader.addAll(featuresArray);
        vectorHeader.add("cardinality");
        produceCsvArrayVectors(vectorHeader, vectors, output, output_delimiter, urlTFPMap);
        return vectors;
    }


    /**
     * Retrieve vectors for DeepSet architecture processing in parallel
     *
     * @param urlQueries       Url for queries file.
     * @param output           Url for output file.
     * @param namespaces       Url for Sparql prefixes file.
     * @param length           Length of queries to get from the csv queries file.
     * @param cores            The count of cores to use in parallel process.
     * @param output_delimiter Delimiter for csv file column.
     * @param input_delimiter  Delimiter for csv file column.
     * @return ArrayList with Map of queries data generated, see @{@link QueryFeatureExtractor}
     */
    public ArrayList<Map<String, Object>> getArrayFeaturesVectorParallel(String urlQueries, String output, String sets, String namespaces, int length, int cores, String output_delimiter, String input_delimiter, String urlTFPMap) {

        model = SparqlUtils.getNamespacesDBPed(namespaces);
        ArrayList<String> featuresArray = new ArrayList<>();
        String[] arraySets = sets.split(",");
        // Add trusted features set to list.
        if (arraySets.length > 1) {
            for (String arraySet : arraySets) {
                if (default_sets.contains(arraySet)) {
                    featuresArray.add(arraySet);
                }
            }
        }
        ArrayList<ArrayList<String>> featInQueryList = this.getArrayQueriesMetaFromCsv(urlQueries, true, input_delimiter, 1, 0, 8, length);
        ForkJoinPool pool = new ForkJoinPool();

        RecursiveDeepSetFeaturizeAction task = new RecursiveDeepSetFeaturizeAction(featInQueryList, featuresArray, cores, output, output_delimiter, urlTFPMap, 0, featInQueryList.size());
        return pool.invoke(task);
    }

    /**
     * Create a csv with array data passed as parameters.
     *
     * @param headers          {@link ArrayList} With header for output file to generate
     * @param list             Queries data list.
     * @param filepath         Output file path.
     * @param indexStart       Index start tupled for build the output filename with de format filename_indexStart_indexLast.csv
     * @param indexLast        Index last tuple for build the output filename with de format filename_indexStart_indexLast.csv
     * @param output_delimiter Delimiter for csv data columns.
     */
    public static void produceCsvArrayVectors(ArrayList<String> headers, ArrayList<Map<String, Object>> list, String filepath, int indexStart, int indexLast, String output_delimiter, String fileTpf) {
        String extension = filepath.substring(filepath.length() - 4);
        produceCsvArrayVectors(
                headers,
                list,
                filepath.substring(0, filepath.length() - 4).concat(String.valueOf(indexStart)).concat("_").concat(String.valueOf(indexLast)).concat(extension),
                output_delimiter,
                fileTpf
        );
    }

    /**
     * Create a csv with array data passed as parameters.
     *
     * @param headers          {@link ArrayList} With header for output file to generate
     * @param list             Queries data list.
     * @param filepath         Output file path.
     * @param output_delimiter Delimiter for csv data columns.
     */
    public static void produceCsvArrayVectors(ArrayList<String> headers, ArrayList<Map<String, Object>> list, String filepath, String output_delimiter, String urlTFPMap) {
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
            if (!urlTFPMap.equals("")) {
                tpfCardinalities = SparqlUtils.readQueryTPFSampling(urlTFPMap,'\t');
            }
            String ENDPOINT = "https://dbpedia.org/sparql";
            int count = 0;
            for (Map<String, Object> queryData : list) {
                for (String header : headers) {
                    switch (header) {
                        case "tables": {
                            ArrayList<String> qTables = (ArrayList<String>) queryData.get("queryTables");
                            if (qTables.size() > 0) {
                                for (String element : qTables) {
                                    sb.append(element);
                                    sb.append(",");
                                }
                            } else {
                                sb.append("EMPTY_VALUE");
                            }
                            break;
                        }
                        case "joins": {
                            ArrayList<String> qTables = (ArrayList<String>) queryData.get("queryJoins");
                            if (qTables.size() > 0) {
                                for (String element : qTables) {
                                    sb.append(element);
                                    sb.append(",");
                                }
                            } else {
                                sb.append("EMPTY_VALUE");
                            }
                            break;
                        }
                        case "joins_v1": {
                            ArrayList<ArrayList> qTables = (ArrayList<ArrayList>) queryData.get("queryJoinsV1Vec");
                            if (qTables.size() > 0) {
                                try {
                                    for (ArrayList<String> elementColOpCard : qTables) {
                                        sb.append(elementColOpCard.get(0));
                                        sb.append(",");
                                        sb.append(elementColOpCard.get(1));
                                        sb.append(",");
                                        sb.append(elementColOpCard.get(2));
                                        sb.append(",");
                                    }
                                } catch (Exception ex) {
                                    ex.printStackTrace();
                                    System.out.println(qTables);
                                }
                            } else {
                                sb.append("EMPTY_VALUE");
                            }
                            break;
                        }
                        case "predicates_v2int": {
                            ArrayList<HashMap<String, Object>> qPredicates = (ArrayList<HashMap<String, Object>>) queryData.get("queryPredicates");
                            if (qPredicates.size() > 0) {
                                for (HashMap<String, Object> element : qPredicates) {
                                    sb.append(element.get("col"));
                                    sb.append(",");
                                    sb.append(element.get("operator"));
                                    sb.append(",");
                                    sb.append(element.get("value"));
                                    sb.append(",");
                                }
                            } else {
                                sb.append("EMPTY_VALUE");
                            }
                            break;
                        }
                        case "predicates_v2uri": {
                            ArrayList<HashMap<String, Object>> qPredicates = (ArrayList<HashMap<String, Object>>) queryData.get("queryPredicatesUris");
                            if (qPredicates.size() > 0) {
                                for (HashMap<String, Object> element : qPredicates) {
                                    sb.append(element.get("col"));
                                    sb.append(",");
                                    sb.append(element.get("operator"));
                                    sb.append(",");
                                    sb.append(element.get("object"));
                                    sb.append(",");
                                }
                            } else {
                                sb.append("EMPTY_VALUE");
                            }
                            break;
                        }
                        case "pred_v2uri_cardinality": {
                            ArrayList<HashMap<String, Object>> qPredicates = (ArrayList<HashMap<String, Object>>) queryData.get("queryPredicatesUris");
                            if (qPredicates.size() > 0) {
                                for (HashMap<String, Object> element : qPredicates) {

                                    sb.append(element.get("col"));
                                    sb.append(",");
                                    sb.append(element.get("operator"));
                                    sb.append(",");
                                    sb.append(element.get("object"));
                                    sb.append(",");
                                    String hash = String.valueOf(element.get("sampling_query_id"));
                                    hash = DigestUtils.md5Hex(hash);
                                    if (tpfCardinalities.get(hash) != null) {
                                        int card = tpfCardinalities.get(hash);
                                        sb.append(card);
                                        sb.append(",");
                                    } else {
                                        String query = String.valueOf(element.get("sampling_query"));
                                        int result = 0;
                                        try {
                                            result = DBPediaUtils.execQuery(query, ENDPOINT, namespases);
                                        } catch (Exception ex) {
                                            ex.printStackTrace();
                                            System.out.println("Query failed: ".concat(hash));
                                        }

                                        sb.append(result);
                                        sb.append(",");
                                        tpfCardinalities.put(hash, result);
                                        sbbrpred_uri.append(hash.concat(" , ".concat(String.valueOf(result))));
                                        sbbrpred_uri.append("\n");
                                    }
                                }
                            } else {
                                sb.append("EMPTY_VALUE");
                            }
                            break;
                        }
                        case "id":
                        case "cardinality": {
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

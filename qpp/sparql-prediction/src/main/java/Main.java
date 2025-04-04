import ged.RDFGraphMatching;
import ged.SparqlUtils;
import semanticweb.sparql.prediction.QueryExecutionTimePredictorMultipleSVR;
import semanticweb.sparql.prediction.TimeClusterer;
import semanticweb.sparql.preprocess.AlgebraFeatureExtractor;
import util.Graph;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.reflect.Array;
import java.util.*;

public class Main {
    public static void main(String[] args){
        //String queryStr = "PREFIX umbel-sc: <http://umbel.org/umbel/sc/> PREFIX linkedgeodata: <http://linkedgeodata.org/triplify/> PREFIX dbpedia: <http://dbpedia.org/ontology/> PREFIX geonames: <http://www.geonames.org/ontology#> PREFIX geo: <http://www.w3.org/2003/01/geo/wgs84_pos#> PREFIX owl: <http://www.w3.org/2002/07/owl#> PREFIX wgs84: <http://www.w3.org/2003/01/geo/wgs84_pos#> PREFIX foaf: <http://xmlns.com/foaf/0.1/> PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX xsd: <http://www.w3.org/2001/XMLSchema#> PREFIX lgdo: <http://linkedgeodata.org/ontology/> PREFIX dbpedia-owl: <http://dbpedia.org/ontology/> PREFIX lgdp: <http://linkedgeodata.org/property/>  PREFIX umbel-sc: <http://umbel.org/umbel/sc/> PREFIX linkedgeodata: <http://linkedgeodata.org/triplify/> PREFIX dbpedia: <http://dbpedia.org/ontology/> PREFIX geonames: <http://www.geonames.org/ontology#> PREFIX geo: <http://www.w3.org/2003/01/geo/wgs84_pos#> PREFIX owl: <http://www.w3.org/2002/07/owl#> PREFIX wgs84: <http://www.w3.org/2003/01/geo/wgs84_pos#> PREFIX foaf: <http://xmlns.com/foaf/0.1/> PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> PREFIX xsd: <http://www.w3.org/2001/XMLSchema#> PREFIX lgdo: <http://linkedgeodata.org/ontology/> PREFIX dbpedia-owl: <http://dbpedia.org/ontology/> PREFIX lgdp: <http://linkedgeodata.org/property/> SELECT DISTINCT ?b ?v0 FROM <http://linkedgeodata.org> WHERE { {?b a lgdo:City} UNION {?b a lgdo:Village} UNION {?b a lgdo:Suburb} UNION {?b a lgdo:Town}.  ?b wgs84:lat ?lat. ?b wgs84:long ?long. OPTIONAL { ?b rdfs:label ?v0 . } } OFFSET 708000 LIMIT 1000";
        //AlgebraFeatureExtractor.test(queryStr); //to get algebra features on a

        /*try {
            Graph g1 = SparqlUtils.buildSPARQL2GXLGraph(queryStr, "1");
            System.out.println(g1);

        } catch (Exception e) {
            System.out.println("Run time exception");;
        }*/
        switch(args[0]){
            case "algebra-extract":
                String path = args[1];
                ArrayList<String> queries = new ArrayList<>();
                ArrayList<String> ids = new ArrayList<>();
                try {
                    Scanner scanner = new Scanner(new File(path));
                    scanner.nextLine();
                    while(scanner.hasNextLine()){
                        String line = scanner.nextLine();
                        String[] splits = line.split(",",2);
                        ids.add(splits[0]);
                        queries.add(splits[1]);
                    }
                } catch (FileNotFoundException e) {
                    throw new RuntimeException(e);
                }
                try {
                    FileWriter writer = new FileWriter(new File(args[2]));
                    writer.write("id,");
                    AlgebraFeatureExtractor z = new AlgebraFeatureExtractor();
                    List<Map.Entry<String, Integer>> ll= new ArrayList<Map.Entry<String, Integer>>(z.getFeatureIndex().entrySet());
                    //fe.featureIndex.entrySet();

                    Collections.sort(ll, new Comparator<Map.Entry<String, Integer>>() {

                        @Override
                        public int compare(Map.Entry<String, Integer> o1,
                                           Map.Entry<String, Integer> o2) {
                            // TODO Auto-generated method stub
                            return o1.getValue()-o2.getValue();
                        }
                    });
                    StringBuilder temp = new StringBuilder();
                    for(Map.Entry<String, Integer> e:ll) {
                        temp.append(e.getKey()).append(",");
                    }
                    temp.deleteCharAt(temp.length()-1);
                    temp.append("\n");
                    writer.write(temp.toString());

                    double[] features;
                    for(int i = 0; i< queries.size(); i++){
                        StringBuilder feat = new StringBuilder();
                        feat.append(ids.get(i));
                        feat.append(",");
                        features = z.extractFeatures(queries.get(i));

                        for(double f: features){
                            feat.append(f);
                            feat.append(",");
                        }
                        feat.deleteCharAt(feat.length()-1);
                        feat.append("\n");
                        writer.write(feat.toString());
                        writer.flush();
                    }
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }


                break;
            case "svm":
                String configFile = System.getenv("CONFIG_FILE");
                try {
                    QueryExecutionTimePredictorMultipleSVR.test(configFile);
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
                break;
            default:
                System.out.println("Choose one of 1) algebra-extract 2) svm\nFor 2) add CONFIG_FILE environmet" +
                        "to configuration file for svm training");
        }

    }
}

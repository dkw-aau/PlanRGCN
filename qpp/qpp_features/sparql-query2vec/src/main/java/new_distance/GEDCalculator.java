package new_distance;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.google.gson.reflect.TypeToken;
import org.apache.commons.lang3.time.StopWatch;
import org.apache.jena.tdb.store.Hash;
import semanticweb.RDFGraphMatching;
import semanticweb.sparql.SparqlUtils;
import util.Graph;

import java.io.*;
import java.lang.reflect.Type;
import java.util.*;
import java.util.concurrent.TimeUnit;

public class GEDCalculator {
    RDFGraphMatching matcher;
    public GEDCalculator(){
        matcher = new RDFGraphMatching();

    }

    public double calculateDistance(String query1, String query2) throws Exception {
        Graph graph1 = SparqlUtils.buildSPARQL2GXLGraph(query1, "query1");
        Graph graph2 = SparqlUtils.buildSPARQL2GXLGraph(query2, "query2");
        return matcher.distanceBipartiteHungarian(graph1,graph2);
    }

    public List<Map<String,String>> readJson(String path) throws FileNotFoundException {
        Type type = new TypeToken<List<Map<String, String>>>() {}.getType();

        Gson gson = new Gson();
        return gson.fromJson(new FileReader(path), type);

    }

    public List<Map<String,String>> makeDistJson(List<Map<String,String>> inputGson){
        List<Map<String,String>> distList = new LinkedList<>();


        for (Map<String,String> i : inputGson){
            Map<String, String> j = new HashMap<>();
            j.put("queryID1",i.get("queryID1"));
            j.put("queryID2",i.get("queryID2"));
            String query1 = i.get("queryString1");
            String query2 = i.get("queryString2");
            String dist = "None";
            StopWatch watch = new StopWatch();
            watch.start();
            try {
                dist = String.valueOf( calculateDistance(query1, query2));

            } catch (Exception e) {
                //dist will become None instead if one of the queries cannot be parsed.
            }
            watch.stop();
            String time = String.valueOf(watch.getTime(TimeUnit.MILLISECONDS));
            j.put("dist",dist);
            j.put("time",time);
            distList.add(j);
        }
        return distList;
    }

    public void writeDist(List<Map<String,String>> distMaps, String outputFile) throws IOException {
        Gson gson = new Gson();
        String jsonText = gson.toJson(distMaps);
        FileWriter fw = new FileWriter(outputFile);
        fw.write(jsonText);
        fw.close();
        System.out.println("Finished " + outputFile);
    }

}

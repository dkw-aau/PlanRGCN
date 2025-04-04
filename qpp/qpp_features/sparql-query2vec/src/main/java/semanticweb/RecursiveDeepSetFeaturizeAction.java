package semanticweb;

import semanticweb.sparql.preprocess.DeepSetFeatureExtractor;
import semanticweb.sparql.preprocess.QueryFeatureExtractor;

import java.util.ArrayList;
import java.util.Map;
import java.util.concurrent.RecursiveTask;

public class RecursiveDeepSetFeaturizeAction extends RecursiveTask<ArrayList<Map<String, Object>>> {

    private ArrayList<ArrayList<String>> queries;
    private int indexStart, indexLast, cores;
    private String output,output_delimiter,urlTpF;
    private ArrayList<String> headerArray;

    public RecursiveDeepSetFeaturizeAction(ArrayList<ArrayList<String>> queries, ArrayList<String> headerArray, int cores, String output,String output_delimiter,String urlTpF, int indexStart, int indexLast) {
        this.queries = queries;
        this.headerArray = headerArray;
        this.cores = cores;
        this.output = output;
        this.output_delimiter = output_delimiter;
        this.urlTpF = urlTpF;
        this.indexStart = indexStart;
        this.indexLast = indexLast;
    }

    private ArrayList<Map<String, Object>> computeFeaturesCreation(int indexStart, int indexLast) {
        ArrayList<String> fail_rows = new ArrayList<>();
        ArrayList<Map<String, Object>> vectors = new ArrayList<>();
        for (int i = indexStart; i < indexLast; i++) {
            try {
                QueryFeatureExtractor qfe = new QueryFeatureExtractor(queries.get(i));
                Map<String, Object> queryVecData = qfe.getProcessedData();
                vectors.add(queryVecData);

            } catch (Exception ex) {
                fail_rows.add(queries.get(i).get(0));
            }
            System.out.println(fail_rows);
        }
        ArrayList<String> vectorHeader = new ArrayList<>();
        vectorHeader.add("id");
        vectorHeader.addAll(headerArray);
        vectorHeader.add("cardinality");
        DeepSetFeatureExtractor.produceCsvArrayVectors(vectorHeader, vectors, output, indexStart, indexLast, output_delimiter,urlTpF);
        return vectors;
    }

    @Override
    protected ArrayList<Map<String, Object>> compute() {
        if (indexLast - indexStart < 50) {
            return computeFeaturesCreation(indexStart, indexLast);
        } else {
            RecursiveDeepSetFeaturizeAction left = new RecursiveDeepSetFeaturizeAction(queries,headerArray, cores, output,output_delimiter,urlTpF, indexStart, (indexLast + indexStart) / 2);
            RecursiveDeepSetFeaturizeAction right = new RecursiveDeepSetFeaturizeAction(queries,headerArray, cores, output,output_delimiter, urlTpF,(indexLast + indexStart) / 2, indexLast);
            left.fork();
            ArrayList<Map<String, Object>> rightAns = right.compute();
            ArrayList<Map<String, Object>> leftAns = left.join();
            leftAns.addAll(rightAns);
            return leftAns;
        }
    }
}
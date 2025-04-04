package ged;

import util.Graph;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.concurrent.RecursiveTask;

public class EditDistanceAction extends RecursiveTask {

    private ArrayList<HashMap<String,Object>> queries;
    private int indexStart, indexLast;
    private int elemtByCore;
    private int cores;
    private String output;

    public EditDistanceAction(ArrayList<HashMap<String,Object>> queries, String output, int cores, int indexStart, int indexLast, int elemtByCore) {
        this.queries = queries;
        this.indexStart = indexStart;
        this.indexLast = indexLast;
        this.elemtByCore = elemtByCore;
        this.output = output;
        this.cores = cores;
    }

    private void computeSubMatrix(int indexStart, int indexLast) {
        StringBuilder sb = new StringBuilder();
        RDFGraphMatching matcher = new RDFGraphMatching();
        StringBuilder failed_row_column = new StringBuilder();
        for (int i = indexStart; i < indexLast; i++) {
            try {
                Graph Gi = (Graph) queries.get(i).get("graph");
                for (int j = 0; j < queries.size(); j++) {
                    if(j==0){
                        sb.append(queries.get(i).get("id"));
                        sb.append(",");
                        sb.append(queries.get(i).get("time"));
                        sb.append(",");
                    }
                    double dist = -1;
                    if (i == j) {
                        sb.append(0.0);
                        sb.append(",");
                        continue;
                    }
                    try{
                        Graph Gj = (Graph) queries.get(j).get("graph");
                        dist =  matcher.distanceBipartiteHungarian(Gi, Gj);
                    }
                    catch (Exception ex){
                       failed_row_column.append(queries.get(j).get("id"));
                       failed_row_column.append("\n");
                    }
                    sb.append(dist);
                    sb.append(",");
                }
                sb.deleteCharAt(sb.length()-1);
                sb.append("\n");
            }
            catch (Exception ex){
                failed_row_column.append(queries.get(i).get("id"));
                failed_row_column.append("\n");
                sb.append("failrow");
                sb.append("\n");
            }
        }
        try {
            BufferedWriter br = new BufferedWriter(new FileWriter(output + "hungarian_distance" + String.format("%06d", indexStart) + "_" + String.format("%06d", indexLast) + ".csv"));
            br.write(sb.toString());
            br.close();
            BufferedWriter br2 = new BufferedWriter(new FileWriter(output + "errors" + String.format("%06d", indexStart) + "_" + String.format("%06d", indexLast) + ".txt"));
            br2.write(failed_row_column.toString());
            br2.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    protected Object compute() {
        if (indexLast - indexStart <= elemtByCore) { // Am I able to process it alone?
            // do the task
            computeSubMatrix(indexStart, indexLast);
            System.out.println("Processed range: "+indexStart+ " - "+indexLast);
        }
        else { // split too big task
            int halfAmount = ((indexLast - indexStart) / 2) + indexStart;
            EditDistanceAction leftTask = new EditDistanceAction(queries, output, cores, indexStart, halfAmount, elemtByCore);
            leftTask.fork(); // add left task to the queue
            EditDistanceAction rightTask = new EditDistanceAction(queries, output, cores, halfAmount, indexLast, elemtByCore);
            rightTask.compute(); // work on right task, this is a recursive call
            leftTask.join(); // wait for queued task to be completed
        }
        return null;
    }
}
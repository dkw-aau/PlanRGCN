package semanticweb;
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
            if(failed_row_column.length()>0){
              BufferedWriter br2 = new BufferedWriter(new FileWriter(output + "errors" + String.format("%06d", indexStart) + "_" + String.format("%06d", indexLast) + ".txt"));
              br2.write(failed_row_column.toString());
              br2.close();
            }


            
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @Override
    protected Object compute() {
        if (indexLast - indexStart <= elemtByCore) { // Am I able to process it alone?
            System.out.println("Beginning on range: "+indexStart+ " - "+indexLast);
            // do the task
            /*if((indexStart==11478 && indexLast==11903) || (indexStart==32735 && indexLast==33160) ||(indexStart==45490&& indexLast==45915) ||(indexStart==36986&& indexLast==37412) ||(indexStart==29759&& indexLast==30184) ||(indexStart==30184&& indexLast==30610) ||(indexStart==12328&& indexLast==12753) ||(indexStart==26783&& indexLast==27209) ||(indexStart==43789&& indexLast==44215) ||(indexStart==47190&& indexLast==47616) ||(indexStart==39963&& indexLast==40388) ||(indexStart==50591&& indexLast==051017) ||(indexStart==31885&& indexLast==32310) ||(indexStart==13178&& indexLast==013604) ||(indexStart==25082&& indexLast==25508) ||(indexStart==43364&& indexLast==43789) ||(indexStart==40388&& indexLast==40814) ||(indexStart==42939&& indexLast==43364) ||(indexStart==31035&& indexLast==31460) ||(indexStart==12753&& indexLast==13178) ||(indexStart == 53568 && indexLast == 53993) ||(indexStart==33585 && indexLast==34011) ||(indexStart==53143 && indexLast==53568) ||(indexStart==53993&& indexLast==54419) ||(indexStart==046765&& indexLast==47190) ||(indexStart==033160&& indexLast==33585) ||(indexStart==25933&& indexLast==26358) ||(indexStart== 26358&& indexLast== 26783) ||(indexStart==031460&& indexLast==31885) ||(indexStart==39538 && indexLast==39963) ||(indexStart==25508 && indexLast==25933) ||(indexStart==036561 && indexLast==36986) ||(indexStart==045065 && indexLast==45490) ||(indexStart==11903 && indexLast==12328)){
                System.out.println("Skipped range: "+indexStart+ " - "+indexLast);
                return null;
            }*/
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
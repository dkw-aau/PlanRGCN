package Benchmark;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Scanner;

public class LSQSampleBenchmark implements Benchmark {
    String filepath;
    List<Query> data = new LinkedList<>();
    public LSQSampleBenchmark(String filepath) {
        this.filepath = filepath;
        Scanner sc = null;
        File f = new File(filepath);

        try {
            sc = new Scanner(f);
        } catch (FileNotFoundException e) {
            throw new RuntimeException(e);

        }
        int iteration = 0;
        while (sc.hasNextLine())
        {
            Scanner linescanner = new Scanner(sc.nextLine());
            linescanner.useDelimiter(",");
            List<String> lineData = new ArrayList<String>();
            linescanner.forEachRemaining(lineData::add);
            if (iteration > 0) {
                Query q = new Query();
                q.id = lineData.get(0);
                q.queryString = lineData.get(1);
                q.type = lineData.get(2);
                q.noNotedTriplePatterns = Integer.parseInt( lineData.get(6));
                q.runTimeMS = Double.parseDouble(lineData.get(7));
                q.resultSize = Integer.parseInt( lineData.get(8));
                data.add(q);
                linescanner.close();
            }
            iteration++;
        }
        sc.close();
    }


    @Override
    public List<Query> getDataset() {
        return data;
    }
}

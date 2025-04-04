package lsq_extract.util;
import com.opencsv.CSVReader;

import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

public class LSQReaderWStats {
    FileReader r;
    CSVReader reader;
    boolean headerTrue;
    public static char delimter = '§';
    public ArrayList<Query> qs = new ArrayList<>();

    public LSQReaderWStats(String path) {
        try  {
            r = new FileReader(path);
            reader = new CSVReader(r, LSQReaderWStats.delimter);
        }catch (Exception e){
            e.printStackTrace();
        }
    }
    
    public ArrayList<Query> readFile() throws IOException {
        String[] record = null;
        reader.readNext();
        while ((record = reader.readNext()) != null){
             String id = record[0];
             String timestamps = record[1];
             String queryString = record[2].replaceAll("\"\"","\"");
             queryString = queryString.replaceAll("\\\\.", ".");
             queryString = queryString.replaceAll("\"\"", "\"");
             double projectVars = Double.parseDouble(record[3]);
             double tripleCount = Double.parseDouble(record[4]);
             double joinVertexCount = Double.parseDouble(record[5]);
             double duration = Double.parseDouble(record[6]);
             Query t = new Query(id, queryString, timestamps, projectVars, tripleCount, joinVertexCount, duration);
             this.qs.add(t);

        }
        return this.qs;
    }

    public ArrayList<Query> getQs(){
        return this.qs;
    }

    public void close(){
        try {
            reader.close();
            r.close();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

    }
}

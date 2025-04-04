package lsq_extract.util;
import com.opencsv.CSVReader;

import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

public class LSQDataReader {
    FileReader r;
    CSVReader reader;
    public ArrayList<String> timestamps;
    public ArrayList<String> queries;
    public ArrayList<String> ids;

    public LSQDataReader(String path) {
        timestamps  = new ArrayList<>()   ;
        queries     = new ArrayList<>()    ;
        ids = new ArrayList<>();
        
        try  {
            r = new FileReader(path);
            reader = new CSVReader(r, '\t');
        }catch (Exception e){
            e.printStackTrace();
        }
    }
    public void readFile() throws IOException {
        String[] record = null;
        reader.readNext();
        while ((record = reader.readNext()) != null){
             ids.add(record[0]);
             timestamps.add(record[1]);
             String queryString = record[2].replaceAll("\"\"","\"");
             queries.add(queryString);
        }
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

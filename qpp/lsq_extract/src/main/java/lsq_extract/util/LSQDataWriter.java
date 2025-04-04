package lsq_extract.util;
import com.opencsv.CSVWriter;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

public class LSQDataWriter {
    File f;
    FileWriter fw;
    CSVWriter writer;
    String[] header = {"queryID","timestamp","queryString"};

    public LSQDataWriter(String path) {
        this.f = new File(path);
        try {
            this.fw= new FileWriter(this.f);
        }catch (IOException e){
            e.printStackTrace();
        }
        this.writer = new CSVWriter(this.fw, '\t',CSVWriter.NO_QUOTE_CHARACTER);
        writer.writeNext(header);


    }


    public  void writeCSV(String id, String timestamp, String queryString){
        writer.writeNext(new String[]{id,timestamp,queryString});
    }

    public void close() throws IOException {
        this.writer.close();
        this.fw.close();
    }
}

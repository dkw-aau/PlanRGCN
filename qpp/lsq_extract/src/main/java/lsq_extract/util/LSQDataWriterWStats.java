package lsq_extract.util;

import com.opencsv.CSVWriter;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

public class LSQDataWriterWStats {
    File f;
    FileWriter fw;
    CSVWriter writer;
    public static String[] header = {"queryID","timestamp","queryString", "projectVariables", "tripleCount", "joinVertexCount", "duration"};
    
    public LSQDataWriterWStats(String path){
        this.f = new File(path);
        try {
            this.fw= new FileWriter(this.f,true);
        }catch (IOException e){
            e.printStackTrace();
        }
        this.writer = new CSVWriter(this.fw, '\t',CSVWriter.NO_QUOTE_CHARACTER);
        writer.writeNext(header);
    }
    public LSQDataWriterWStats(String path, Boolean writeHeader){
        this.f = new File(path);
        try {
            this.fw= new FileWriter(this.f,true);
        }catch (IOException e){
            e.printStackTrace();
        }
        this.writer = new CSVWriter(this.fw, '\t',CSVWriter.NO_QUOTE_CHARACTER);
        if(writeHeader){
            writer.writeNext(header);
        }
    }

    
    public  void writeCSV(String id, String timestamp, String queryString, String projectVariables, String tripleCount, String joinVertexCount, String duration)throws IOException{
        writer.writeNext(new String[]{id,timestamp,queryString,projectVariables, tripleCount, joinVertexCount, duration}, false);
        this.writer.flush();
    }
    public  void writeCSV(String id, String queryString, String duration) throws IOException{
        writer.writeNext(new String[]{id,queryString, duration}, false);
        this.writer.flush();
    }
    public  void writeCSV(String id, String queryString, String duration, String rc) throws IOException{
        writer.writeNext(new String[]{id,queryString, duration, rc}, false);
        this.writer.flush();
    }
    public void setHeader(String[] head){
        LSQDataWriterWStats.header = head;
    }

    public void close() throws IOException {
        
        this.writer.close();
        this.fw.close();
    }
}

package lsq_extract.benchmark;

import com.opencsv.CSVWriter;

import lsq_extract.util.Query;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

public class QueryLogger {
    File f;
    FileWriter fw;
    CSVWriter writer;
    public static boolean newDataFormat = false;
    public static char delimiter = '§';
    public static String[] header = {"queryID","timestamp","queryString", "projectVariables", "tripleCount", "joinVertexCount", "resultSize","latency", "duration"};
    String[] algebraOps = new String[]{
        "triple", "bgp", "join", "leftjoin", "union", "filter", "graph", "extend", "minus", "path*",
        "pathN*", "path+", "pathN+", "path?", "notoneof", "tolist", "order", "project", "distinct", "reduced",
        "multi", "top", "group", "assign", "sequence", "slice", "treesize"};
    String[] completeHeader = new String[QueryLogger.header.length + this.algebraOps.length];
    
    public QueryLogger(String path) throws IOException{
        this.f = new File(path);
        this.fw= new FileWriter(this.f);
        this.writer = new CSVWriter(this.fw, QueryLogger.delimiter,CSVWriter.NO_QUOTE_CHARACTER);
        
        System.arraycopy(QueryLogger.header, 0, this.completeHeader, 0, QueryLogger.header.length);
        System.arraycopy(this.algebraOps, 0, this.completeHeader, QueryLogger.header.length, this.algebraOps.length);
        writer.writeNext(this.completeHeader);
    }

    public void logQueries(ArrayList<Query> qs) throws IOException{
        logQueries(qs,QueryLogger.newDataFormat);
    }

    public void logQueries(ArrayList<Query> qs, boolean isNewFormat) throws IOException{
        if (isNewFormat){
            logQueriesNewFormat(qs);
        }else{
            for(Query i: qs){
                String[] data = new String[this.completeHeader.length];
                data[0] = i.id;
                data[1] = i.timestamp;
                data[2] = i.text;
                data[3] = String.valueOf( i.projectVars);
                data[4] = String.valueOf(i.tripleCount);
                data[5] = String.valueOf(i.joinVertexCount);
                data[6] = String.valueOf(i.resultSize);
                data[7] = String.valueOf(i.latency);
                data[8] = String.valueOf(i.duration);
           
                for(int idx = 9; idx < this.completeHeader.length; idx++){
                    data[idx] = String.valueOf( i.getOp(this.completeHeader[idx]));
                }
                writer.writeNext(data);

            }
            this.writer.close();
            this.fw.close();
        }
    }

    private void logQueriesNewFormat(ArrayList<Query> qs) throws IOException {
        for(Query i: qs){
            String[] data = new String[this.completeHeader.length];
            data[0] = i.id;
            data[1] = i.text;
            data[2] = String.valueOf(i.duration);
            data[3] = i.resultSize == -1 ? null : String.valueOf(i.resultSize);
       
            for(int idx = 4; idx < this.completeHeader.length; idx++){
                data[idx] = String.valueOf( i.getOp(this.completeHeader[idx]));
            }
            writer.writeNext(data);

        }
        this.writer.close();
        this.fw.close();
    }
    
}

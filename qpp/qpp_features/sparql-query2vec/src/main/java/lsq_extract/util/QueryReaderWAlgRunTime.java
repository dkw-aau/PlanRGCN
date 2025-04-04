package lsq_extract.util;
import com.opencsv.CSVReader;

import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;

// reads the output of the query logger.
public class QueryReaderWAlgRunTime {
    FileReader r;
    CSVReader reader;
    boolean headerTrue;
    public static boolean extra = false;
    public static boolean parseQueryString = true;
    public static boolean newLSQdata = false;
    public ArrayList<Query> qs = new ArrayList<>();

    public QueryReaderWAlgRunTime(String path) {
        try  {
            r = new FileReader(path);
            reader = new CSVReader(r, '\t'); // replace with _ in linux
        }catch (Exception e){
            e.printStackTrace();
        }
    }
    public ArrayList<Query> readFile() throws IOException{
        return readFile(QueryReaderWAlgRunTime.newLSQdata);
    }
    public ArrayList<Query> readFile(boolean newFormat) throws IOException {
        String[] record = null;
        String[] req = reader.readNext();
        
        while ((record = reader.readNext()) != null){
            if(record.length < 2){
                continue;
            } 
            Query t = readLine(record, req, QueryReaderWAlgRunTime.newLSQdata);
             
            this.qs.add(t);

        }
        return this.qs;
    }

    public ArrayList<Query> getQs(){
        return this.qs;
    }
    public ArrayList<String[]> getQueryData(){
        ArrayList<String[]> qs= new ArrayList<>();
        for (Query i: this.qs){
            qs.add(new String[]{i.id,i.text, String.valueOf( i.duration)});
        }
        return qs;
    }


    public Query readLine(String[] record,String[] header, boolean newDataformat){
        if(extra){
            return readLineIdQuery(record);
        }
        if (newDataformat){
            return readLineNewData(record);
        }else{
            return readLineOldData(record,header);
        }
    }
    private Query readLineOldData(String[] record, String[] header) {
        String id = record[0];
        String timestamps = record[1];
        String queryString = parseQueryStr(record[2]);
        double projectVars = Double.parseDouble(record[3]);
        double tripleCount = Double.parseDouble(record[4]);
        double joinVertexCount = Double.parseDouble(record[5]);
        int resultSize = Integer.parseInt(record[6]);
        double latency = Double.parseDouble(record[7]);
        double duration = Double.parseDouble(record[8]);


        Query t = new Query(id, queryString, timestamps, projectVars, tripleCount, joinVertexCount, duration);
        for(int i = 9; i<record.length; i++){
                t.addOp(header[i], Double.parseDouble( record[i]));
             }
        t.resultSize = resultSize;
        t.latency = latency;
        return t;

    }
    public Query readLineNewData(String[ ] record){
        if(record.length <=2){ }
        System.out.println(record);
        String id = record[0];
        String queryString = record[1]; //parseQueryStr(record[1]);
        double duration = Double.parseDouble(record[2]);
        int resultSize = record[3].isBlank() ? -1 : (int) Double.parseDouble(record[3]);
        Query t = new Query(id, queryString, "", -1, -1, -1, duration);
        t.resultSize = resultSize;
        return t;
    }
    public Query readLineIdQuery(String[] record){
        String id = record[0];
        String queryString = record[1];
        Query t;
        try{
        t = new Query(id, queryString, Double.parseDouble( record[11] ) );
        }catch(java.lang.ArrayIndexOutOfBoundsException e){
        t = new Query(id, queryString, Double.parseDouble( record[2]) );
        }

        return t;
    }

    public String parseQueryStr(String query){
        if (QueryReaderWAlgRunTime.parseQueryString){
            String t = query.replaceAll("\"\"","\"");
            t = t.replaceAll("\\\\.", ".");
            //queryString = queryString.replaceAll("\"\"", "\"");
            t = t.replaceAll("\s\"\s", "\s\"\"\s");
            t = t.replaceAll("\s\"[)]", "\s\"\")");
            return t;
        }else
            return query;
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

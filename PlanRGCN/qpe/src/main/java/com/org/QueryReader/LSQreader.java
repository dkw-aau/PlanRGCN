package com.org.QueryReader;

import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

import com.opencsv.CSVParserBuilder;
import com.opencsv.CSVReader;
import com.opencsv.CSVReaderBuilder;

public class LSQreader {
    String path;
    LinkedList<String> queries = new LinkedList<>();
    LinkedList<String> ids = new LinkedList<>();
    CSVReader reader;

    public LinkedList<String> getQueries() {
        return queries;
    }
    public LinkedList<String> getIds() {
        return ids;
    }

    public LSQreader(String file_path) throws FileNotFoundException{
        path = file_path;
        //CSVReader reader = new CSVReader(new FileReader(App.class.getClassLoader().getResource("csv.csv").getFile()), ',','"','-');
        CSVReader c = new CSVReaderBuilder(new FileReader(file_path))
                .withCSVParser(new CSVParserBuilder()
                        .withQuoteChar('\"')
                        .withSeparator('\t')
                        .build())
                .build();
        //CSVParser parser = new CSVParser(c);
        reader = c;
        try{
            c.readNext();
        
        }catch (Exception e){
            e.printStackTrace();
        }
    }

    public ArrayList<String> getNext(){
        try{
            ArrayList<String> query_id = new ArrayList<>();
            String [] nextLine;
            if ((nextLine = reader.readNext()) != null) {
                
                query_id.add(nextLine[0]);
                query_id.add(nextLine[1]);
                return query_id;
            }else{
                return null;
            }
        
        }catch (Exception e){
            e.printStackTrace();
        }
        return null;
        
    }
}

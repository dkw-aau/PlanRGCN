package lsq_extract.util;

import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import com.opencsv.CSVParserBuilder;
import com.opencsv.CSVReader;
import com.opencsv.CSVReaderBuilder;

public class QueryReaderLegacy extends QueryReaderWAlgRunTime {
    public static boolean isWDbench = false;
    public QueryReaderLegacy(String path) {
        super(path);
        this.qs = new ArrayList<>();
        try {
            // Create a CSVReader for TSV format
            CSVReader c = new CSVReaderBuilder(new FileReader(path))
                .withCSVParser(new CSVParserBuilder()
                        .withQuoteChar('\"')
                        .withSeparator('\t')
                        .build())
                .build();

                try{
                    List<String[]> allRows = c.readAll();
                    Iterator<String[]> iter = allRows.iterator();
                    iter.next();
                    while( iter.hasNext()){
                        String[] i = iter.next();
			lsq_extract.util.Query q;
                        if(isWDbench){
			   q = new Query(i[1]);
                           q.id = i[0];
                           q.duration = -1.0;
                           q.resultSize = -1;
			}else{
			   q = new Query(i[0]);
                           q.id = i[1];
                           q.duration = Double.valueOf( i[2]);
                           q.resultSize = -1;
			}
			this.qs.add(q);
                    }
                
                }catch (Exception e){
                    e.printStackTrace();
                }

            reader.close(); // Close the reader when done
        } catch (IOException e) {
            System.err.println("Error reading the TSV file: " + e.getMessage());
        }

        //TODO Auto-generated constructor stub
    }
}

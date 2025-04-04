package lsq_extract.util;
import org.apache.jena.query.ARQ;
import org.apache.jena.query.Query;
import org.apache.jena.query.QueryFactory;
import org.apache.jena.query.QueryParseException;
import org.apache.jena.sparql.algebra.Algebra;

import java.io.IOException;

public class LegalQueryChecker {
    LSQDataWriter legalQueries;
    LSQDataWriter illegalQueries;

    public LegalQueryChecker(String legalpath, String illegalpath) {
        legalQueries = new LSQDataWriter(legalpath);
        illegalQueries = new LSQDataWriter(illegalpath);
        ARQ.init();
    }
    public LegalQueryChecker() {
        ARQ.init();
    }
    public void legalQueryProcessor(LSQDataReader reader){
        int total = 0;
        int legal = 0;
        int illegal=0;
        for (int i =0; i<reader.queries.size();i++){
            total++;
            if(checkQuery(reader.queries.get(i))){
                legal++;
                legalQueries.writeCSV(reader.ids.get(i), reader.timestamps.get(i),reader.queries.get(i));
            }else{
                illegal++;
                illegalQueries.writeCSV(reader.ids.get(i), reader.timestamps.get(i),reader.queries.get(i));
            }
        }
        try {
            legalQueries.close();
            illegalQueries.close();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        System.out.println("Total amount of queries: "+ total);
        System.out.println("Valid Queries: "+legal);
        System.out.println("Invalid Queries "+illegal);

    }
    public static boolean checkQuery(String queryString){
        try {
            Query q = QueryFactory.create(queryString);
            Algebra.compile(q);
        }catch (QueryParseException e){
            return false;
        }
        return true;
    }
}

package semanticweb.sparql.preprocess;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

public abstract  class FeaturesExtractorBase {

    /**
     * Retrieve list of queries tuples  in csv dataset file.
     *
     * @param url         Url fil csv with queries info.
     * @param header      If csv include header
     * @param delimiter   Delimiter character
     * @param queryColumn Csv column that contain query string( Csv must contain other data)
     * @param idColumn    Csv column that contain the query id
     * @param length      Length of the queries with cardinality  to load > zero
     * @return a list of Queries in format [idQuery,Query,execTimeColumn]
     */
    public ArrayList<ArrayList<String>> getArrayQueriesMetaFromCsv(String url, boolean header, String delimiter, int queryColumn, int idColumn, int execTimeColumn, int length) {
        return  this.getArrayQueriesMetaFromCsv(url, header, delimiter, queryColumn, idColumn, execTimeColumn,-1 , length);
    }
    /**
     * Retrieve list of queries tuples  in csv dataset file.
     *
     * @param url         Url fil csv with queries info.
     * @param header      If csv include header
     * @param delimiter   Delimiter character
     * @param queryColumn Csv column that contain query string( Csv must contain other data)
     * @param idColumn    Csv column that contain the query id
     * @param length      Length of the queries with cardinality  to load > zero
     * @return a list of Queries in format [idQuery,Query,execTimeColumn,Cardinality]
     */
    public ArrayList<ArrayList<String>> getArrayQueriesMetaFromCsv(String url, boolean header, String delimiter, int queryColumn, int idColumn, int execTimeColumn, int cardinalityColumn, int length) {
        String row;
        ArrayList<ArrayList<String>> arrayList = new ArrayList<>();
        int countQueryInProcess = 0;
        //Al menos 3 columnas son requeridas(id,query,time)
        int csvColumns = 3;
        try {
            BufferedReader csvReader = new BufferedReader(new FileReader(url));
            if (header) {
                //Ignore first read that corresponde with header
                String headerCols = csvReader.readLine();
                csvColumns = headerCols.split(",").length;
            }
            // if length is equal to zero not restrict the length of queries.
            while ((row = csvReader.readLine()) != null && (countQueryInProcess < length || length == 0)) {
                String[] rowArray = row.split(delimiter);

                if (csvColumns == 3)
                    csvColumns = rowArray.length;

                while (rowArray.length > csvColumns) {
                    //join query column that contain delimiter symbol in one string.
                    ArrayList<String> listTemp = new ArrayList<>();
                    for (int i = 0; i < rowArray.length ; i++) {
                        if(i < queryColumn){
                            listTemp.add(i, rowArray[i]);
                        }
                        else if(i == queryColumn){
                            String first = rowArray[queryColumn];
                            String second = rowArray[queryColumn + 1];
                            listTemp.add(queryColumn, first.concat(delimiter).concat(second));
                        }
                        else if (i != queryColumn + 1) {
                            listTemp.add(i- 1, rowArray[i]);
                        }
                    }
                    rowArray = listTemp.toArray(new String[0]);
                }

                row = rowArray[queryColumn];
                //Remove quotes in init and end of the string...
                row = row.replaceAll("^\"|\"$", "");
                ArrayList<String> predicatesAndId = new ArrayList<>();
                if (idColumn >= 0)
                    predicatesAndId.add(rowArray[idColumn]);
                predicatesAndId.add(row);
                predicatesAndId.add(rowArray[execTimeColumn]);
                //Add cardinality column inf index > 0
                if (cardinalityColumn >= 0)
                    predicatesAndId.add(rowArray[cardinalityColumn]);
                try {
                    if (Integer.parseInt(rowArray[execTimeColumn]) > 0) {
                        countQueryInProcess++;
                    } else {
                        // If cardinality not > 0 not add the query to list.
                        continue;
                    }
                } catch (Exception ex) {
                    ex.printStackTrace();
                    continue;
                }

                arrayList.add(predicatesAndId);
            }
            csvReader.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return arrayList;
    }
}

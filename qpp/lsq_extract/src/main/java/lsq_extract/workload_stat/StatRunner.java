package lsq_extract.workload_stat;

import java.io.IOException;
import java.util.ArrayList;

import lsq_extract.benchmark.QueryLogger;
import lsq_extract.util.Query;
import lsq_extract.util.QueryReaderWAlgRunTime;

public class StatRunner {
    String queryPath;
    //LSQReaderWStats reader;
    QueryReaderWAlgRunTime reader;

    public StatRunner(String queryPath){
        this.queryPath = queryPath;
        
    }

    public void RunStat(String outputpath) throws IOException{
        //this.reader = new LSQReaderWStats(this.queryPath);
        QueryReaderWAlgRunTime.newLSQdata = true;
        QueryReaderWAlgRunTime.parseQueryString = false;
        this.reader = new QueryReaderWAlgRunTime(this.queryPath);
        
        ArrayList<Query> qs = this.reader.readFile();
        OperatorStat stat = new OperatorStat(qs);
        qs = stat.addOperatorStats();
        QueryLogger.delimiter = '\t';
        QueryLogger.header = new String[]{"queryID","queryString","duration","resultCount"};
        QueryLogger.newDataFormat = true;
        QueryLogger ql = new QueryLogger(outputpath);
        ql.logQueries(qs);
        //qs = stat.addOperatorStats();
        //this.reader.qs = qs;
        
    }
    
    /*public LSQReaderWStats getReader(){
        return this.reader;
    }*/
    public QueryReaderWAlgRunTime getReader(){
        return this.reader;
    }
}

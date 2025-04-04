package lsq_extract.benchmark;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.util.ArrayList;

import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import lsq_extract.util.LSQReaderWStats;
import lsq_extract.util.Query;
import lsq_extract.workload_stat.StatRunnerTest;

public class QueryExecutorTest {

    //should be disabled if the endpoint is not running.
    @Disabled
    @Test
    void testExecuteQuery() {
        String filetoread = StatRunnerTest.class.getResource("data.csv").toString();
        filetoread = filetoread.substring(5);
        LSQReaderWStats reader = new LSQReaderWStats(filetoread);
        ArrayList<Query> qs = null;
        try {
            qs = reader.readFile();
        } catch (IOException e) {
            assertTrue(false);
        }
        assertNotNull(qs);
        QueryExecutor ex = new QueryExecutor("http://172.21.233.23:8890/sparql/", qs);
        ex.benchmarkQueries();
        assertNotNull(qs.get(0).latency);
        System.out.println(qs.get(0).latency +" : "+ qs.get(0).resultSize);
        assertTrue(qs.get(0).latency > 0);
        assertTrue(qs.get(0).resultSize > 1);
    }
}

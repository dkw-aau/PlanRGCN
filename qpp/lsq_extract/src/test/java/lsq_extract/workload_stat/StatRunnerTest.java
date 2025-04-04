package lsq_extract.workload_stat;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.util.ArrayList;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import lsq_extract.util.LSQReaderWStats;
import lsq_extract.util.Query;

@Disabled
public class StatRunnerTest {
    ArrayList<Query> qs = null;

    /*@BeforeEach
    public void init(){
        String filetoread = StatRunnerTest.class.getResource("data.csv").toString();
        filetoread = filetoread.substring(5);
        LSQReaderWStats reader = new LSQReaderWStats(filetoread);
        try {
            qs = reader.readFile();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }*/

    @Test
    @Disabled
    public void testRunStat() {
        String filetoread = StatRunnerTest.class.getResource("data.csv").toString();
        filetoread = filetoread.substring(5);
        StatRunner runner = new StatRunner(filetoread);
        try {
            runner.RunStat(filetoread.substring(0,filetoread.length()-8) +"/output.csv");
        } catch (IOException e) {
            assertTrue(false);
        }
        assertNotNull(runner.getReader().qs);
        assertEquals(13,runner.getReader().qs.get(0).getOp("triple"),0.1);

    }
}

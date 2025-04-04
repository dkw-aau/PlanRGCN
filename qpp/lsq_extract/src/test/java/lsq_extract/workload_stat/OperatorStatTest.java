package lsq_extract.workload_stat;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import lsq_extract.util.*;

@Disabled
public class OperatorStatTest {
    
    @Test
    public void test01(){
        String testQuery = "SELECT  * WHERE   { ?city     <http://ww.google.com/foo#test>    <http://ww.google.com/foo#place> . }";
        lsq_extract.util.Query q = new Query(testQuery);
        ArrayList<lsq_extract.util.Query> qs = new ArrayList<>();
        qs.add(q);
        OperatorStat stat = new OperatorStat(qs);
        stat.addOperatorStats();
        assertEquals(1, q.getOp("triple"));
        assertEquals(1, q.getOp("bgp"));
    
    }
}

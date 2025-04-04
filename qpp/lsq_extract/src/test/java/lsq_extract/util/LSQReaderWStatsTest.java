package lsq_extract.util;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.util.ArrayList;

@Disabled 
public class LSQReaderWStatsTest {
    //TODO: need redo of actual values.
    @Disabled
    @Test
    public void Test01(){
        String filetoread = LSQReaderWStatsTest.class.getResource("data.csv").toString();
        filetoread = filetoread.substring(5);
        LSQReaderWStats reader = new LSQReaderWStats(filetoread);
        ArrayList<Query> qs = null;
        try {
            qs = reader.readFile();
        } catch (IOException e) {
            e.printStackTrace();
        }
        assertNotNull(qs);
        assertTrue(qs.size() > 0);
        assertEquals(qs.get(0).id, "http://lsq.aksw.org/lsqQuery-A1sqWvmzIcD4FrZfN0GcooA4EEAl1QfDvK_sgN-Wg9g");
        assertEquals(qs.get(0).timestamp, "2010-07-20T02:00:00Z^^http://www.w3.org/2001/XMLSchema#dateTime");
        assertEquals(qs.get(0).tripleCount, 1,0.01);
        assertEquals(qs.get(0).joinVertexCount, 0, 0.01);
    }
}

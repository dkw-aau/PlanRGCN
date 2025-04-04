package new_distance;

import org.apache.jena.shacl.lib.G;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class GEDCalculatorTest {
    GEDCalculator calculator;
    File i;

    @BeforeEach
    void setUp() {
        calculator = new GEDCalculator();
        i = new File("src/test/resources/new_distance/test.json");
    }

    @Test
    void calculateDistance() {

        assertTrue(i.exists());
        try {
            List<Map<String,String>> map = calculator.readJson("src/test/resources/new_distance/test.json");
            assertEquals(map.get(0).get("queryID1"),"queryid1");
            assertEquals(map.get(0).get("queryString2"),"SELECT * {?s ?p <http://foo2>}");
            assertEquals(map.get(1).get("queryString2"),"SELECT * {?s ?p <http://foo6>}");
        } catch (FileNotFoundException e) {
            throw new RuntimeException(e);
        }
    }


    @Test
    void makeDistJson01() {
        try {
            List<Map<String,String>> map = calculator.readJson("src/test/resources/new_distance/test.json");
            List<Map<String,String>> distMap = calculator.makeDistJson(map);
            assertTrue(distMap.get(0).containsKey("dist"));
            assertTrue(distMap.get(0).containsKey("time"));
            assertTrue(distMap.get(1).containsKey("dist"));
        } catch (FileNotFoundException e) {
            throw new RuntimeException(e);
        }
    }

    @Test
    void writeDist() {
        try {
            List<Map<String,String>> map = calculator.readJson("src/test/resources/new_distance/test.json");
            List<Map<String,String>> distMap = calculator.makeDistJson(map);
            calculator.writeDist(distMap, "src/test/resources/new_distance/testWDist.json");
            File file = new File("src/test/resources/new_distance/testWDist.json");
            assertTrue(file.exists());
            if (file.exists()){
                file.delete();
            }
        } catch (FileNotFoundException e) {
            throw new RuntimeException(e);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
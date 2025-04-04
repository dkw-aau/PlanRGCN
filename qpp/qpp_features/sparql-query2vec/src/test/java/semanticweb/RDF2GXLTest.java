package semanticweb;

import org.junit.jupiter.api.Test;
import util.Graph;

class RDF2GXLTest {
    

    @Test
    void transformTriple2GXL() {
    }

    @Test
    void testTransformTriple2GXL() {
    }

    @Test
    void getGXLRootElement() {
    }

    @Test
    void getGXLGraphElement() {
    }

    @Test
    void getTestGXLGraph() {
        try {
            Graph testGXLGraph = RDF2GXL.getTestGXLGraph();
            System.out.println(testGXLGraph);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}
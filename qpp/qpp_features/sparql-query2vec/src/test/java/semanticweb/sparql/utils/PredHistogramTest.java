package semanticweb.sparql.utils;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Disabled;
@Disabled
public class PredHistogramTest {
    @Disabled
    @Test
    public void testCreatePredHistogram() {
        PredHistogram predHistogram = new PredHistogram("uri","sub"," Select a....","rdf:type");
        assertEquals("", predHistogram.getOnSubQuery()," Select a....");
        assertEquals("",predHistogram.getOnSubType(),"uri");
    }
    @Disabled
    @Test
    public void testToSamplingQueryFileString() {
        PredHistogram predHistogram = new PredHistogram("numeric","sub"," Sub a....","rdf:type");
        String resp = predHistogram.toSamplingQueryFileString(",");
        //assertEquals("", 1,resp.split("\n").length);

        predHistogram.setQuery("obj","uri","Obj a....");
        resp = predHistogram.toSamplingQueryFileString(",");
        //assertEquals("", 2,resp.split("\n").length);
        assertEquals("",predHistogram.getOnObjType(),"numeric");

        PredHistogram predHistogram1 = new PredHistogram("uri","obj"," Obj a....","rdf:type");
        resp = predHistogram1.toSamplingQueryFileString(",");
        //assertEquals("", 1,resp.split("\n").length);
    }
}

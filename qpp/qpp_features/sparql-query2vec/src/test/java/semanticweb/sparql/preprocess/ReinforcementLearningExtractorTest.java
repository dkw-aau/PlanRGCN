package semanticweb.sparql.preprocess;

import java.io.IOException;
import java.util.ArrayList;

import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import lsq_extract.util.QueryReaderWAlgRunTime;
import lsq_extract.util.Query;
@Disabled
public class ReinforcementLearningExtractorTest {
    @Disabled
    @Test
    public void testGetArrayFeaturesVector() {
        String filetoread = ReinforcementLearningExtractorTest.class.getResource("sample.csv").toString();
        filetoread = filetoread.substring(5);
        lsq_extract.util.QueryReaderWAlgRunTime reader = new QueryReaderWAlgRunTime(filetoread);
        try {
            reader.readFile();
        } catch (IOException e) {
            assertTrue(false);
        }
        reader.close();
        ArrayList<lsq_extract.util.Query> qs =reader.qs;
        assertTrue(qs.get(0).id.equals( "http://lsq.aksw.org/lsqQuery-nXQBU_Ghcbptmr4XhKLGnru-IGX7krpAEp_jqeM-jK8"));
        assertEquals(0.002383057,qs.get(0).duration, 0.01);
        
        String out_path = ReinforcementLearningExtractorTest.class.getResource("sample.csv").toString();
        out_path = out_path.substring(5,out_path.length()-10);
        out_path = out_path +"extraTest";
        ReinforcementLearningExtractor re = new ReinforcementLearningExtractor();
        re.getArrayFeaturesVector(qs,out_path,"ᶷ");
    }
}

package lsq_extract.util;
import org.json.simple.JSONObject;

import java.io.IOException;
import java.io.StringWriter;

public class Triples {
    String subject;
    String predicate;
    String object;

    public Triples(String subject, String predicate, String object) {
        this.subject = subject;
        this.predicate = predicate;
        this.object = object;
    }

    @Override
    public String toString() {
        JSONObject obj = new JSONObject();
        obj.put("subject",subject);
        obj.put("predicate",predicate);
        obj.put("object",object);
        StringWriter out = new StringWriter();
        try {
            obj.writeJSONString(out);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        return out.toString();
    }
}

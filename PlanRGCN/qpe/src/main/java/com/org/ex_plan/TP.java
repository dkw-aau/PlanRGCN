package com.org.ex_plan;

import java.util.ArrayList;

public class TP extends Node {
    public String subject;
    public String predicate;
    public String object;
    public String nodeType;

    public boolean isLiteral;
    public String objectDatatype;
    public String objectLang;
    
    public TP(String subject, String predicate) {
        this.subject = subject;
        this.predicate = predicate;
        nodeType = "TP";
    }
    public TP(String subject2) {
        this.subject = subject2;
        nodeType = "TP";
    }
    @Override
    public String getJoinedVar(String term){
        if(term.equals(this.subject)){
            return "S";
        }
        if(term.equals(this.predicate)){
            return "P";
        }
        if(term.equals(this.object)){
            return "O";
        }
        return null;
    }

    @Override
    public String toString() {
        return this.subject + " + " + this.predicate + " + "+ this.object;
    }


    @Override
    public ArrayList<String> getTerms() {
        ArrayList<String> terms = new ArrayList<>();
        terms.add(subject);
        terms.add(predicate);
        terms.add(object);
        return terms;
    }
    
}

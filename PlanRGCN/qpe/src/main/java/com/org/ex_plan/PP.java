package com.org.ex_plan;

import java.util.ArrayList;

public class PP extends TP {
    ArrayList<String> predicateList;
    int pred_min;
    int pred_max;
    ArrayList<String> pathComplexity; //pathOperator
    
    public PP(String subject) {
        super(subject);
        nodeType = "PP";
    }

    @Override
    public String getJoinedVar(String term) {
        if(term.equals(this.subject)){
            return "S";
        }
        if( predicateList.contains(term) ){
            return "P";
        }
        if(term.equals(this.object)){
            return "O";
        }
        return null;
    }

    @Override
    public ArrayList<String> getTerms() {
        ArrayList<String> terms = new ArrayList<>();
        terms.add(subject);
        terms.addAll(predicateList);
        terms.add(object);
        return terms;
    }
    
}

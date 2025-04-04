package com.org.ex_plan;

import java.util.ArrayList;

public abstract class Node {
    int nodeId;
    
    public abstract String getJoinedVar(String term);
    public abstract ArrayList<String> getTerms();
}

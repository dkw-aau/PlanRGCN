package com.org.ex_plan;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedList;

public class BinaryOperator {
    public String opName;
    int status = 0; // 0 mean left, 1 right, 2 means done
    public LinkedList<Node> rightNodes = new LinkedList<>();
    public LinkedList<Node> leftNodes = new LinkedList<>();
    
    
    public ArrayList<Edge> edges = new ArrayList<>();
    HashMap<String, ArrayList<Node>> joinMap = new HashMap<>();

    public BinaryOperator(String name){
        opName = name;
    }
    
    public int getStatus() {
        return status;
    }

    public void startRight(){
        status = 1;
    }

    public void done(){
        status = 2;
    }
    public boolean isRight(){
        if (status == 1)
            return true;
        return false;
    }

    public boolean isLeft(){
        if (status == 0)
            return true;
        return false;
    }

    public void addNode(Node node){
        if(isRight()){
            for (String term : node.getTerms()){
                if(joinMap.keySet().contains(term)){
                    for(Node i : joinMap.get(term)){
                        edges.add(new Edge(node, i, opName));
                    }
                }
            }
            rightNodes.add(node);
        }else if(isLeft()){
            for (String term : node.getTerms()){
                if(joinMap.keySet().contains(term)){
                    joinMap.get(term).add(node);
                }else{
                    ArrayList<Node> nlst = new ArrayList<>();
                    nlst.add(node);
                    joinMap.put(term, nlst);
                }
            }
            leftNodes.add(node);
        }
        //Nothing should happen if the operator is neither in right or left state.
        /*else{
            throw new RuntimeException();
        }*/
    }
    
}

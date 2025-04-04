package com.org.ex_plan;

public class Edge {
    Node node1;
    Node node2;
    String edge_type;
    
    public Edge(Node node1, Node node2, String edge_type) {
        this.node1 = node1;
        this.node2 = node2;
        this.edge_type = edge_type;
    }

    @Override
    public String toString() {
        return "Edge: Node " + node1.nodeId + " + Node" + node2.nodeId + "  +  " + edge_type;
    }
    
}

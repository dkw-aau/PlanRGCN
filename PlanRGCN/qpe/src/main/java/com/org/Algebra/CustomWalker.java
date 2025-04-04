package com.org.Algebra;

import java.util.Iterator;

import org.apache.jena.sparql.algebra.Op;
import org.apache.jena.sparql.algebra.OpVisitor;
import org.apache.jena.sparql.algebra.OpVisitorByType;
import org.apache.jena.sparql.algebra.op.Op0;
import org.apache.jena.sparql.algebra.op.Op1;
import org.apache.jena.sparql.algebra.op.Op2;
import org.apache.jena.sparql.algebra.op.OpExt;
import org.apache.jena.sparql.algebra.op.OpFilter;
import org.apache.jena.sparql.algebra.op.OpLeftJoin;
import org.apache.jena.sparql.algebra.op.OpN;

public class CustomWalker extends OpVisitorByType {
        private final OpVisitor   beforeVisitor;
        private final OpVisitor   afterVisitor;
        protected final OpVisitor visitor;

        public CustomWalker(OpVisitor visitor, OpVisitor beforeVisitor, OpVisitor afterVisitor) {
            this.visitor = visitor;
            this.beforeVisitor = beforeVisitor;
            this.afterVisitor = afterVisitor;
        }

        public CustomWalker(OpVisitor visitor) {
            this(visitor, null, null);
        }

        protected final void before(Op op) {
            if ( beforeVisitor != null )
                op.visit(beforeVisitor);
        }

        protected final void after(Op op) {
            if ( afterVisitor != null )
                op.visit(afterVisitor);
        }

        @Override
        protected void visit0(Op0 op) {
            before(op);
            if ( visitor != null )
                op.visit(visitor);
            after(op);
        }

        @Override
        protected void visit1(Op1 op) {
            before(op);
            if ( visitor != null )
                op.visit(visitor);
            if ( op.getSubOp() != null )
                op.getSubOp().visit(this);
            after(op);
        }

        @Override
        protected void visitFilter(OpFilter op) {
            visit1(op);
        }

        @Override
        protected void visitLeftJoin(OpLeftJoin op) {
            visit2(op);
        }

        @Override
        protected void visit2(Op2 op) {
            before(op);
            if ( visitor != null )
                op.visit(visitor);
            if ( op.getLeft() != null )
                op.getLeft().visit(this);
            if ( op.getRight() != null )
                op.getRight().visit(this);
            after(op);
        }

        @Override
        protected void visitN(OpN op) {
            before(op);
            if ( visitor != null )
                op.visit(visitor);
            for (Iterator<Op> iter = op.iterator(); iter.hasNext();) {
                Op sub = iter.next();
                sub.visit(this);
            }
            after(op);
        }

        @Override
        protected void visitExt(OpExt op) {
            before(op);
            if ( visitor != null )
                op.visit(visitor);
            if ( op.effectiveOp() != null )
                // Walk the effective op, if present.
                op.effectiveOp().visit(this);
            after(op);
        }
    }
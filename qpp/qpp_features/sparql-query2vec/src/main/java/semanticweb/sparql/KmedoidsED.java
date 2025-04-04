package semanticweb.sparql;

import com.clust4j.algo.AbstractCentroidClusterer;
import com.clust4j.algo.KMedoidsParameters;
import com.clust4j.algo.LabelEncoder;
import com.clust4j.except.IllegalClusterStateException;
import com.clust4j.log.LogTimer;
import com.clust4j.log.Log.Tag.Algo;
import com.clust4j.metrics.pairwise.Distance;
import com.clust4j.metrics.pairwise.GeometricallySeparable;
import com.clust4j.utils.VecUtils;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.TreeMap;
import java.util.Map.Entry;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.util.FastMath;

public final class KmedoidsED extends AbstractCentroidClusterer {
    private static final long serialVersionUID = -4468316488158880820L;
    public static final GeometricallySeparable DEF_DIST;
    public static final int DEF_MAX_ITER = 10;
    private volatile int[] medoid_indices;
    private volatile double[][] dist_mat;
    private volatile TreeMap<Integer, Double> med_to_wss;

    protected KmedoidsED(RealMatrix data,RealMatrix distances) {
        this(data, distances, 5);
    }

    protected KmedoidsED(RealMatrix data,RealMatrix distances, int k) {
        this(data,distances,(new KMedoidsParameters(k)).setMetric(new EditDistanceSimilarity(data)));
    }

    protected KmedoidsED(RealMatrix data, RealMatrix distances, KMedoidsParameters planner) {
        super(data, planner);
        this.medoid_indices = new int[this.k];
        this.dist_mat = distances.getData();
        this.med_to_wss = new TreeMap();
        if (!this.dist_metric.equals(Distance.MANHATTAN)) {
            this.warn("KMedoids is intented to run with Manhattan distance, WSS/BSS computations will be inaccurate");
        }

    }

    public String getName() {
        return "KMedoids";
    }

    protected KmedoidsED fit() {
        synchronized(this.fitLock) {
            if (null != this.labels) {
                return this;
            } else {
                LogTimer timer = new LogTimer();
                double[][] X = this.data.getData();
                double nan = 0.0D / 0.0;
                if (1 == this.k) {
                    this.labelFromSingularK(X);
                    this.fitSummary.add(new Object[]{this.iter, this.converged, this.tss, this.tss, this.tss, 0.0D / 0.0, timer.wallTime()});
                    this.sayBye(timer);
                    return this;
                } else {
                    this.dist_mat = data.getData();
                    this.info("distance matrix computed in " + timer.toString());
                    this.medoid_indices = this.init_centroid_indices;
                    int[] newMedoids = this.medoid_indices;
                    double bestCost = 1.0D / 0.0;
                    double maxCost = -1.0D / 0.0;
                    double avgCost = 0.0D / 0.0;
                    double wss_sum = 0.0D / 0.0;
                    boolean convergedFromCost = false;

                    for(boolean configurationChanged = true; configurationChanged && this.iter < this.maxIter; configurationChanged = !this.converged) {
                        KmedoidsED.ClusterAssignments clusterAssignments;
                        try {
                            clusterAssignments = this.assignClosestMedoid(newMedoids);
                        } catch (IllegalClusterStateException var26) {
                            this.exitOnBadDistanceMetric(X, timer);
                            return this;
                        }

                        KmedoidsED.MedoidReassignmentHandler rassn;
                        try {
                            rassn = new KmedoidsED.MedoidReassignmentHandler(clusterAssignments);
                        } catch (IllegalClusterStateException var25) {
                            this.exitOnBadDistanceMetric(X, timer);
                            return this;
                        }

                        if (rassn.new_clusters.size() == 1) {
                            this.k = 1;
                            this.warn("(dis)similarity metric cannot partition space without propagating Infs. Returning one cluster");
                            this.labelFromSingularK(X);
                            this.fitSummary.add(new Object[]{this.iter, this.converged, this.tss, this.tss, this.tss, 0.0D / 0.0, timer.wallTime()});
                            this.sayBye(timer);
                            return this;
                        }

                        newMedoids = rassn.reassignedMedoidIdcs;
                        boolean lastIteration = VecUtils.equalsExactly(newMedoids, this.medoid_indices);
                        this.converged = lastIteration || (convergedFromCost = FastMath.abs(wss_sum - bestCost) < this.tolerance);
                        double tmp_wss_sum = rassn.new_clusters.total_cst;
                        double tmp_bss = this.tss - tmp_wss_sum;
                        if (tmp_wss_sum > maxCost) {
                            maxCost = tmp_wss_sum;
                        }

                        if (tmp_wss_sum < bestCost) {
                            wss_sum = tmp_wss_sum;
                            bestCost = tmp_wss_sum;
                            this.labels = rassn.new_clusters.assn;
                            this.med_to_wss = rassn.new_clusters.costs;
                            this.centroids = rassn.centers;
                            this.medoid_indices = newMedoids;
                            this.bss = tmp_bss;
                            avgCost = tmp_wss_sum / (double)this.k;
                        }

                        if (this.converged) {
                            this.reorderLabelsAndCentroids();
                        }

                        this.fitSummary.add(new Object[]{this.iter, this.converged, this.tss, avgCost, wss_sum, this.bss, timer.wallTime()});
                        ++this.iter;
                    }

                    if (!this.converged) {
                        this.warn("algorithm did not converge");
                    } else {
                        this.info("algorithm converged due to " + (convergedFromCost ? "cost minimization" : "harmonious state"));
                    }

                    this.sayBye(timer);
                    return this;
                }
            }
        }
    }

    private void exitOnBadDistanceMetric(double[][] X, LogTimer timer) {
        this.warn("distance metric (" + this.dist_metric + ") produced entirely equal distances");
        this.labelFromSingularK(X);
        this.fitSummary.add(new Object[]{this.iter, this.converged, this.tss, this.tss, this.tss, 0.0D / 0.0, 0.0D / 0.0, timer.wallTime()});
        this.sayBye(timer);
    }

    private KmedoidsED.ClusterAssignments assignClosestMedoid(int[] medoidIdcs) {
        boolean all_tied = true;
        int[] assn = new int[this.m];
        double[] costs = new double[this.m];

        for(int i = 0; i < this.m; ++i) {
            boolean is_a_medoid = false;
            double minDist = 1.0D / 0.0;
            int nearest = -1;
            int[] var12 = medoidIdcs;
            int var13 = medoidIdcs.length;

            for(int var14 = 0; var14 < var13; ++var14) {
                int medoid = var12[var14];
                if (i == medoid) {
                    nearest = medoid;
                    minDist = this.dist_mat[i][i];
                    is_a_medoid = true;
                    break;
                }

                int rowIdx = FastMath.min(i, medoid);
                int colIdx = FastMath.max(i, medoid);
                if (this.dist_mat[rowIdx][colIdx] < minDist) {
                    minDist = this.dist_mat[rowIdx][colIdx];
                    nearest = medoid;
                }
            }

            if (-1 == nearest) {
                nearest = medoidIdcs[this.getSeed().nextInt(this.k)];
            }

            if (!is_a_medoid) {
                all_tied = false;
            }

            assn[i] = nearest;
            costs[i] = minDist;
        }

        if (all_tied) {
            throw new IllegalClusterStateException("entirely stochastic process: all distances are equal");
        } else {
            return new KmedoidsED.ClusterAssignments(assn, costs);
        }
    }

    public Algo getLoggerTag() {
        return Algo.KMEDOIDS;
    }

    protected Object[] getModelFitSummaryHeaders() {
        return new Object[]{"Iter. #", "Converged", "TSS", "Avg Clust. Cost", "Min WSS", "Max BSS", "Wall"};
    }

    protected void reorderLabelsAndCentroids() {
        LabelEncoder encoder = (new LabelEncoder(this.labels)).fit();
        this.labels = encoder.getEncodedLabels();
        int i = 0;
        this.centroids = new ArrayList();
        int[] classes = encoder.getClasses();
        int[] var4 = classes;
        int var5 = classes.length;

        for(int var6 = 0; var6 < var5; ++var6) {
            int claz = var4[var6];
            this.centroids.add(this.data.getRow(claz));
            this.wss[i++] = (Double)this.med_to_wss.get(claz);
        }

    }

    protected final GeometricallySeparable defMetric() {
        return DEF_DIST;
    }

    static {
        DEF_DIST = Distance.MANHATTAN;
    }

    private class ClusterAssignments extends TreeMap<Integer, ArrayList<Integer>> {
        private static final long serialVersionUID = -7488380079772496168L;
        final int[] assn;
        TreeMap<Integer, Double> costs;
        double total_cst;

        ClusterAssignments(int[] assn, double[] costs) {
            this.assn = assn;
            this.costs = new TreeMap();

            for(int i = 0; i < assn.length; ++i) {
                int medoid = assn[i];
                double cost = costs[i];
                ArrayList<Integer> ref = (ArrayList)this.get(medoid);
                if (null == ref) {
                    ref = new ArrayList();
                    ref.add(i);
                    this.put(medoid, ref);
                    this.costs.put(medoid, cost);
                } else {
                    ref.add(i);
                    double d = (Double)this.costs.get(medoid);
                    this.costs.put(medoid, d + cost);
                }

                this.total_cst += cost;
            }

        }
    }

    private class MedoidReassignmentHandler {
        final KmedoidsED.ClusterAssignments init_clusters;
        final ArrayList<double[]> centers;
        final int[] reassignedMedoidIdcs;
        final KmedoidsED.ClusterAssignments new_clusters;

        MedoidReassignmentHandler(KmedoidsED.ClusterAssignments assn) {
            this.centers = new ArrayList(KmedoidsED.this.k);
            this.reassignedMedoidIdcs = new int[KmedoidsED.this.k];
            this.init_clusters = assn;
            this.medoidAssn();
            this.new_clusters = KmedoidsED.this.assignClosestMedoid(this.reassignedMedoidIdcs);
        }

        void medoidAssn() {
            int i = 0;

            for(Iterator var3 = this.init_clusters.entrySet().iterator(); var3.hasNext(); ++i) {
                Entry<Integer, ArrayList<Integer>> pair = (Entry)var3.next();
                ArrayList<Integer> members = (ArrayList)pair.getValue();
                double minCost = 1.0D / 0.0;
                int bestMedoid = 0;
                Iterator var12 = members.iterator();

                while(var12.hasNext()) {
                    int a = (Integer)var12.next();
                    double medoidCost = 0.0D;
                    Iterator var14 = members.iterator();

                    while(var14.hasNext()) {
                        int b = (Integer)var14.next();
                        if (a != b) {
                            int rowIdx = FastMath.min(a, b);
                            int colIdx = FastMath.max(a, b);
                            medoidCost += KmedoidsED.this.dist_mat[rowIdx][colIdx];
                        }
                    }

                    if (medoidCost < minCost) {
                        minCost = medoidCost;
                        bestMedoid = a;
                    }
                }

                this.reassignedMedoidIdcs[i] = bestMedoid;
                this.centers.add(KmedoidsED.this.data.getRow(bestMedoid));
            }

        }
    }
}


package semanticweb.sparql;

import com.clust4j.metrics.pairwise.GeometricallySeparable;
import org.apache.commons.math3.linear.RealMatrix;

public class EditDistanceSimilarity implements GeometricallySeparable {

    private RealMatrix distances;

    public EditDistanceSimilarity(RealMatrix distances) {
        this.distances = distances;
    }

    public RealMatrix getDistances() {
        return distances;
    }

    @Override
    public double getDistance(double[] doubles, double[] doubles1) {
        return 0;
    }

    @Override
    public double getPartialDistance(double[] doubles, double[] doubles1) {
        return 0;
    }

    @Override
    public double partialDistanceToDistance(double v) {
        return 0;
    }

    @Override
    public double distanceToPartialDistance(double v) {
        return 0;
    }

    @Override
    public String getName() {
        return "EDIT DISTANCE";
    }
}

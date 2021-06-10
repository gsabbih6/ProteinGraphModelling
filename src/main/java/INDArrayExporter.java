import org.jgrapht.Graph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.PrintStream;
import java.io.PrintWriter;
import java.util.Iterator;

public class INDArrayExporter {
    int featuresize = 0;
    String label = "";

    public INDArrayExporter(int featuresize, String label) {
        this.featuresize = featuresize;
        this.label = label;
    }

    public Protein getProteinAsAX(Graph<AminoAcid, Bond> g, boolean selfloop) {
        return new Protein(adjacencyMatrix(g, selfloop), features(g), label());
    }

    private INDArray label() {
        INDArray l = Nd4j.zeros(1, 2);
        if (label.equalsIgnoreCase("enzyme"))
            return l.putScalar(0, 0, 1);
        return l.putScalar(0, 1, 1);
    }

    private INDArray features(Graph<AminoAcid, Bond> g) { //uses a onehot encoding for now
        int n = g.vertexSet().size();
        INDArray fMatrix = Nd4j.zeros(n, featuresize);
        Iterator vertexIt1 = g.vertexSet().iterator();
        System.out.println("Graph Size= " + n + "x" + n);
        int i = 0;
        while (vertexIt1.hasNext()) { // looping through each vertex ie N rows
            AminoAcid v = (AminoAcid) vertexIt1.next();
            int id = Integer.parseInt(Constants.getAANumber(v));
            fMatrix.putScalar(i, id, 1);
            i++;
        }
        return fMatrix;
    }

    private INDArray adjacencyMatrix(Graph<AminoAcid, Bond> g, boolean selfLoop) {
        int n = g.vertexSet().size();
        INDArray mMatrix = Nd4j.zeros(n, n);
        Iterator vertexIt1 = g.vertexSet().iterator();
        System.out.println("Graph Size= " + n + "x" + n);
        int i = 0;
        while (vertexIt1.hasNext()) { // looping through each vertex ie N rows
            AminoAcid v = (AminoAcid) vertexIt1.next();
            Iterator vertexIt2 = g.vertexSet().iterator();
            int j = 0;
            while (vertexIt2.hasNext()) { //looping through each vertex ie N columns
                AminoAcid u = (AminoAcid) vertexIt2.next();
                Bond e = g.getEdge(v, u);
                if (v.equals(u) && selfLoop) {
                    mMatrix.putScalar(i, j, 1);
                } else if (e != null) {
                    mMatrix.putScalar(i, j, 1);
                }
                j++;
            }
            i++;
        }
        return mMatrix;
    }

}
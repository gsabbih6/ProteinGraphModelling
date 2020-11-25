import org.jgrapht.Graph;
import org.jgrapht.nio.GraphExporter;

import java.io.OutputStream;
import java.io.PrintStream;
import java.io.PrintWriter;
import java.io.Writer;
import java.util.Iterator;

public class CustomCSVExporter {
    int featuresize = 0;

    public CustomCSVExporter(int featuresize) {
        this.featuresize = featuresize;
    }

    public void exportGraph(Graph<AminoAcid, Bond> g, PrintStream writer) {

        int n = g.vertexSet().size();
        Iterator vertexIt1 = g.vertexSet().iterator();
        PrintWriter out = new PrintWriter(writer);
        System.out.println("Graph Size is 0 " + n + "x" + (n + featuresize) + " that is each vertices has ");
        while (vertexIt1.hasNext()) { // looping through each vertex ie N rows

            AminoAcid v = (AminoAcid) vertexIt1.next();
            Iterator vertexIt2 = g.vertexSet().iterator();
            int i = 0;
            while (vertexIt2.hasNext()) { //looping through each vertex ie N columns
                AminoAcid u = (AminoAcid) vertexIt2.next();
                Bond e = g.getEdge(v, u);
                if (v.equals(u)) {
                    System.out.print("1"); // self loops
                } else

                    if (e == null) {
//                    this.exportEscapedField(out, "0");
                    System.out.print("0");
                } else {
                    System.out.print("1");
//                    this.exportEscapedField(out, "1");
                }

                if (i++ < n - 1) {
//                    out.print(',');
                    System.out.print(',');
                }
            }

            // add X feature vector
            int id = Integer.parseInt(Constants.getAANumber(v));
            System.out.print(',');
            for (int j = 0; j < featuresize; j++) {
//                System.out.println(j);
                if (j == id) {
                    System.out.print("1");
//                    this.exportEscapedField(out, "1");
                } else {
                    System.out.print("0");
//                    this.exportEscapedField(out, "0");
                }
                if (j < featuresize - 1) {
//                    out.print(',');
                    System.out.print(',');
                }
            }

//            out.println();
            System.out.println();

        }


    }

    private void exportEscapedField(PrintWriter out, String field) {
//        System.out.print(field + ',');
        out.print(Utils.escapeDSV(field, ','));
    }
}

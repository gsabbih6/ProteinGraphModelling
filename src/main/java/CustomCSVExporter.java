import org.jgrapht.Graph;
import org.jgrapht.nio.GraphExporter;

import java.io.OutputStream;
import java.io.PrintWriter;
import java.io.Writer;
import java.util.Iterator;

public class CustomCSVExporter {
    int featuresize = 0;

    public CustomCSVExporter(int featuresize) {
        this.featuresize = featuresize;
    }

    //        @Override
    public void exportGraph(Graph<AminoAcid, Bond> g, Writer writer) {
        int n = g.vertexSet().size();
        Iterator vertexIt1 = g.vertexSet().iterator();
        PrintWriter out = new PrintWriter(writer);
        while (vertexIt1.hasNext()) { // looping through each vertex ie N rows

            AminoAcid v = (AminoAcid) vertexIt1.next();
            Iterator vertexIt2 = g.vertexSet().iterator();
            int i = 0;
            while (vertexIt2.hasNext()) { //looping through each vertex ie N columns
                AminoAcid u = (AminoAcid) vertexIt2.next();
                Bond e = g.getEdge(v, u);
                if (e == null) {
                    this.exportEscapedField(out, "0");
                } else {
                    this.exportEscapedField(out, "1");
                }

                if (i++ < n - 1) {
                    out.print(',');
                }
            }

            // add X feature vector
            int id = Constants.getAANumber(v);
            for (int j = 0; j < featuresize; j++) {

                if (j == id) {
                    this.exportEscapedField(out, "1");
                } else {
                    this.exportEscapedField(out, "0");
                }
                if (j++ < featuresize - 1) {
                    out.print(',');
                }
            }

            out.println();

        }


    }

    private void exportEscapedField(PrintWriter out, String field) {
        out.print(Utils.escapeDSV(field, ','));
    }
}

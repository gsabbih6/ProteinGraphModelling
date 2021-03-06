import com.mxgraph.layout.*;
import com.mxgraph.model.mxCell;
import com.mxgraph.swing.mxGraphComponent;
import com.mxgraph.util.mxConstants;
import org.jgrapht.ext.JGraphXAdapter;
import org.jgrapht.graph.DefaultUndirectedWeightedGraph;
import org.jgrapht.alg.clique.*;
import org.jgrapht.nio.Attribute;
import org.jgrapht.nio.DefaultAttribute;
import org.jgrapht.nio.dot.DOTExporter;
import org.nd4j.linalg.api.ndarray.INDArray;
//import org.jgrapht.nio.matrix.MatrixExporter;

import javax.swing.*;
import java.awt.*;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.util.*;
import java.util.List;

public class ProteinGraph extends DefaultUndirectedWeightedGraph<AminoAcid, Bond> {
    public static final String EXPORT_TYPE_CSV = "csv";
    public static final String EXPORT_TYPE_MATRIX = "matrix";
    public static final String EXPORT_TYPE_DOT = "dot";
    public static final String EXPORT_TYPE_IMAGE = "image";
    private static final long serialVersionUID = 2202072534703043194L;
    private JGraphXAdapter<AminoAcid, Bond> jGraphXAdapter;
    private static final Dimension DEFAULT_SIZE = new Dimension(530, 320);
    private String filename;

    public ProteinGraph(Class<? extends Bond> edgeClass) {
        super(edgeClass);
    }

    public void exportGraph(String supportedType, String filename, String label) throws FileNotFoundException {
        this.filename = filename;
        switch (supportedType) {
            case EXPORT_TYPE_CSV:
                exportAsCSV(this, label);
                break;
            case EXPORT_TYPE_MATRIX:
                exportAsMatrix(this, label);
                break;
            case EXPORT_TYPE_DOT:
                exportAsDOT(this);
                break;
            case EXPORT_TYPE_IMAGE:
                exportAsImage();
                break;
            default:

        }

    }

    private void exportAsImage() {
//        mxPngImageEncoder
    }

    private void exportAsMatrix(ProteinGraph proteinGraph, String label) throws FileNotFoundException {
//        new MatrixExporter().exportAdjacencyMatrix( new PrintWriter(out), proteinGraph );
        PrintWriter out = new PrintWriter(filename + ".txt");
        new CustomCSVExporter(25, label).exportGraph(proteinGraph, out, " ");
    }

    private void exportAsCSV(ProteinGraph proteinGraph, String label) throws FileNotFoundException {
        // new CustomCSVExporter(25).exportGraph(proteinGraph, System.out);
        System.out.println(filename);
        PrintWriter out = new PrintWriter(filename + ".csv");
        new CustomCSVExporter(25, label).exportGraph(proteinGraph, out, ",");

    }

    public Protein getProteinAsAX(String label) {
//        return new INDArrayExporter(1,"a").adjacencyMatrix(this,true);
        return new INDArrayExporter(26, label).getProteinAsAX(this, true);
    }


    private void exportAsDOT(ProteinGraph proteinGraph) throws FileNotFoundException {
        DOTExporter<AminoAcid, Bond> exporter =
                new DOTExporter<>();
        exporter.setVertexAttributeProvider((v) -> {
            Map<String, Attribute> map = new LinkedHashMap<>();
            map.put("label", DefaultAttribute.createAttribute(v.getLabel()));
            return map;
        });
        //Writer writer = new StringWriter();
        PrintWriter writer = new PrintWriter(filename + ".dot");
        exporter.exportGraph(proteinGraph, writer);
//        System.out.println(writer.toString());
    }

    public Set<Set<AminoAcid>> findMaximumCliques() {
        // this may be time consuming based on the size of the graph
        Set<Set<AminoAcid>> result = new LinkedHashSet<>();
        BronKerboschCliqueFinder<AminoAcid, Bond> alg = new BronKerboschCliqueFinder<>(this);

        if (!alg.isTimeLimitReached()) {
            Iterator<Set<AminoAcid>> it = alg.maximumIterator();
            while (it.hasNext()) {
                result.add(it.next());
            }
        }
        return result;
    }

    public Set<Set<AminoAcid>> findMaximumCliques(long timeout,
                                                  java.util.concurrent.TimeUnit uni) {
        // this may be time consuming based on the size of the graph
        Set<Set<AminoAcid>> result = new LinkedHashSet<>();
        BronKerboschCliqueFinder<AminoAcid, Bond> alg = new BronKerboschCliqueFinder<>(this, timeout, uni);

        if (alg.isTimeLimitReached()) {
            Iterator<Set<AminoAcid>> it = alg.maximumIterator();
            while (it.hasNext()) {
                result.add(it.next());
            }
        }
        return result;
    }

    public JFrame visualize(String title) {
        // show network mapGraph
        if (this.vertexSet().size() > 0) {
            jGraphXAdapter = new JGraphXAdapter<>(this);
//            jGraphXAdapter.setLabelsClipped(true);
//            jGraphXAdapter.set

            jGraphXAdapter.getModel().beginUpdate();
            jGraphXAdapter.clearSelection();
            jGraphXAdapter.selectAll();
            jGraphXAdapter.setCellsEditable(false);
            jGraphXAdapter.setCellsDisconnectable(false);
            jGraphXAdapter.setEdgeLabelsMovable(false);
            jGraphXAdapter.setCellsResizable(false);
            Object[] cells = jGraphXAdapter.getSelectionCells();
//            createStyles(jGraphXAdapter);
            for (Object c : cells) {
                mxCell cell = (mxCell) c;
                if (((mxCell) c).isVertex()) {
                    AminoAcid a = (AminoAcid) cell.getValue();
                    cell.setValue(a.getLabel());//a.getLabel() + "(" + a.getId() + ")");
                    System.out.println(a.getLabel());
                    cell.getGeometry().setWidth(40);
                    cell.getGeometry().setHeight(40);
                    Map<String, Object> nodeStyle = jGraphXAdapter.getCellStyle(cell);
                    nodeStyle.put(mxConstants.STYLE_SHAPE, mxConstants.SHAPE_ELLIPSE);
//                    nodeStyle.put(mxConstants.STYLE_FILLCOLOR, Constants.colormap().get(a.getLabel()));
                    jGraphXAdapter.getStylesheet().putCellStyle("a", nodeStyle);
//                    jGraphXAdapter.getModel().setStyle(cell,
//                            "{"+mxConstants.STYLE_FILLCOLOR + "=" +
//                            Constants.colormap().get(a.getLabel())+","+mxConstants.STYLE_SHAPE + "=" +
//                                    mxConstants.SHAPE_ELLIPSE+"}");
                    jGraphXAdapter.getModel().setStyle(cell,
                            mxConstants.STYLE_FILLCOLOR + "="
                                    +
                                    Constants.colormap().get(a.getLabel())
                    );
//                     jGraphXAdapter.getModel().setStyle(cell,nodeStyle.toString());
//                    m = new mxGeometry();
//                    jGraphXAdapter.getModel().getStyle(cell);

//                    cell.setStyle("a");a
//                    cell.setStyle(nodeStyle.toString());
//                    System.out.println(jGraphXAdapter.getAlternateEdgeStyle());
                } else {
                    Bond a = (Bond) cell.getValue();
                    cell.setValue("");
                    if (a.getLabel().equalsIgnoreCase(Bond.HYDROGEN_BOND)) {
                        jGraphXAdapter.getModel().setStyle(cell,
                                mxConstants.STYLE_STROKECOLOR + "="
                                        +
                                        "#9da3a1");
                        jGraphXAdapter.getModel().setStyle(cell,
                                mxConstants.STYLE_DASHED + "="
                                        +
                                        "true");
                    }
                }
            }

            mxIGraphLayout layout = new mxCircleLayout(jGraphXAdapter);

            layout.execute(jGraphXAdapter.getDefaultParent());

            jGraphXAdapter.getModel().endUpdate();

            mxGraphComponent graphComponent = new mxGraphComponent(jGraphXAdapter);
//            graphComponent.setConnectable(false);
//            graphComponent.getGraph().setCellStyle("Circle");//setAllowDanglingEdges(false);
//            ge

            JFrame frame = new JFrame();
            frame.setTitle(title);
            frame.setSize(new Dimension(800, 600));
            frame.getContentPane().add(new JScrollPane(graphComponent));
            frame.pack();
            frame.setVisible(true);
            return frame;
        }
        return null;
    }

    public void addAAwithPeptideBonds(Map<Integer, AminoAcid> aminoAcids) {
        int i = 0;
        int j = 1;
        int seqL = aminoAcids.size();
        List<AminoAcid> lList = new LinkedList<>(aminoAcids.values());
        while (i < seqL - 1 || j < seqL) {
            this.addVertex(lList.get(i));
            this.addVertex(lList.get(j));
            this.addEdge(lList.get(i), lList.get(j), new Bond(Bond.PEPTIDE_BOND, String.valueOf(i)));
            i++;
            j++;
        }

    }

    public void addHydrogenBonds(Map<AminoAcid, ArrayList<AminoAcid>> aminoAcidsHBonds) {
        // Add HBonds
        int i = 0;
        for (Map.Entry<AminoAcid, ArrayList<AminoAcid>> es : aminoAcidsHBonds.entrySet()) {

            AminoAcid aa = es.getKey();
            if (aa == null) continue;
            for (AminoAcid aaa : es.getValue()) {


                this.addEdge(aa, aaa, new Bond(Bond.HYDROGEN_BOND, String.valueOf(i)));
                i++;

            }
        }

//
    }
//
//    private void createStyles(JGraphXAdapter<AminoAcid, Bond> jGraphXAdapter) {
//
//        for (Map.Entry<String, String> s : Constants.colormap().entrySet()) {
//            Map<String, Object> nodeStyle = new Hashtable<>();
//            nodeStyle.put(mxConstants.STYLE_SHAPE, mxConstants.SHAPE_ELLIPSE);
//            nodeStyle.put(mxConstants.STYLE_FILLCOLOR, s.getValue());
////            jGraphXAdapter.getce
//            System.out.println(nodeStyle);
//        }
//
//
////        Map<String, Object> sty = jGraphXAdapter.getModel().st;
////        sty.put(mxConstants.STYLE_SHAPE, mxConstants.SHAPE_ELLIPSE);
////
////        System.out.println();
////        jGraphXAdapter.getStylesheet().putCellStyle("mys", sty);
//    }
}

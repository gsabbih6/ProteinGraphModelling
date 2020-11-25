import java.io.FileNotFoundException;
import java.util.*;

public class MainClass {

    public static void main(String[] args) throws FileNotFoundException {

        Preprocess preprocess = new Preprocess("1fd4", "A");
        Map<Integer, AminoAcid> aminoAcids = preprocess.getAminoAcidSet();//
        Map<AminoAcid, ArrayList<AminoAcid>> aminoAcidsHBonds = preprocess.getBondArrayListMap();//

//        System.out.println(preprocess.printDSSPFormat());
        ProteinGraph proteinGraph = new ProteinGraph(Bond.class);
        proteinGraph.addAAwithPeptideBonds(aminoAcids);////
        proteinGraph.addHydrogenBonds(aminoAcidsHBonds);//

//        System.out.println(proteinGraph.findMaximumCliques());
//        proteinGraph.visualize("Visualizer");
//        proteinGraph.exportGraph(ProteinGraph.EXPORT_TYPE_DOT);
        proteinGraph.exportGraph(ProteinGraph.EXPORT_TYPE_CSV);//
    }
}

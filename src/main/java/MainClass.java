import java.util.*;

public class MainClass {

    public static void main(String []args) {
        Preprocess preprocess = new Preprocess("1EOT", "A");
        preprocess.process();
        Map<Integer, AminoAcid> aminoAcids = preprocess.getAminoAcidSet();
        Map<AminoAcid, ArrayList<AminoAcid>> aminoAcidsHBonds = preprocess.getBondArrayListMap();


        ProteinGraph proteinGraph = new ProteinGraph(Bond.class);
//        for (AminoAcid a : aminoAcids) {
//            proteinGraph.addVertex(a);
//        }

        // add peptide Bond
        int i = 0;
        int j = 1;
        int seqL = aminoAcids.size();
        List<AminoAcid> lList = new LinkedList<>(aminoAcids.values());
        while (i < seqL - 1 || j < seqL) {
//            AminoAcid a1 = lList.get(i);
//            AminoAcid a2 = lList.get(j);
            proteinGraph.addVertex(lList.get(i));
            proteinGraph.addVertex(lList.get(j));
            proteinGraph.addEdge(lList.get(i), lList.get(j), new Bond(Bond.PEPTIDE_BOND, String.valueOf(i)));
            i++;
            j++;
//            System.out.println(i + " " + j);
        }

        // Add HBonds
        i = 0;
        for (Map.Entry<AminoAcid, ArrayList<AminoAcid>> es : aminoAcidsHBonds.entrySet()) {

            AminoAcid aa = es.getKey();
            for (AminoAcid aaa : es.getValue()) {

                proteinGraph.addEdge(aa, aaa, new Bond(Bond.HYDROGEN_BOND, String.valueOf(i)));
                i++;

            }
        }


        System.out.println(proteinGraph.findMaximumCliques());
        proteinGraph.visualize("Hi");
        proteinGraph.exportGraph(ProteinGraph.EXPORT_TYPE_DOT);
//        proteinGraph.exportGraph(ProteinGraph.);
    }
}

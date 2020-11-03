import org.biojava.nbio.structure.*;
import org.biojava.nbio.structure.align.util.AtomCache;
import org.biojava.nbio.structure.contact.AtomContactSet;
import org.biojava.nbio.structure.secstruc.SecStrucCalc;
import org.biojava.nbio.structure.secstruc.SecStrucInfo;
import org.biojava.nbio.structure.secstruc.SecStrucState;

import java.io.IOException;
import java.io.Serializable;
import java.util.*;

public class Preprocess {

    private final String PDBID;
    Map<Integer, AminoAcid> aminoAcidSet;
    Map<AminoAcid, ArrayList<AminoAcid>> hBondArrayListMap;
    private Structure structure;
    private String proteinChain;

    public Preprocess(String PBDID, String proteinChain) {
        this.PDBID = PBDID;
        this.proteinChain = proteinChain;
        aminoAcidSet = new LinkedHashMap<>();
        hBondArrayListMap = new LinkedHashMap<>();
        setStructure(PBDID);
//        setSecondaryStructure(PBDID);
    }

    private void setSecondaryStructure(List<Group> groups) {
//        //        for (Chain c : structure.getChains()) {
//        for (Group g : groups) {
//            if (g.hasAminoAtoms()) { //Only AA store SS
//                //Obtain the object that stores the SS
//                SecStrucInfo ss = (SecStrucState) g.getProperty(Group.SEC_STRUC);
//                //Print information: chain+resn+name+SS
//                System.out.println(" " +
//                        g.getResidueNumber() + " " +
//                        g.getPDBName() + " -> " + ss.getAssignment());
//            }
//        }
////        }

//        Predict and assign the SS of the Structure
        SecStrucCalc ssp = new SecStrucCalc(); //Instantiation needed
        try {
            List<SecStrucState> secondaryStructure = ssp.calculate(structure, true); //true assigns the SS to the Structure

            for (SecStrucState state : secondaryStructure) {
                Group g = state.getGroup();
                if (g.getChain().getName().equalsIgnoreCase(proteinChain)) {

//                    The correct strength of such hydrogen bonds is experimentally known to vary greatly from
//                    ≈5–6 kcal/mol for the isolated bond to ≈0.5–1.5 kcal/mol for proteins in solution
//
//                    Energetics of hydrogen bonds in peptides
//                    Sheh-Yi Sheu, Dah-Yen Yang, H. L. Selzle, and E. W. Schlag
                    int resid = g.getResidueNumber().getSeqNum();
                    int res1 = resid == state.getAccept1().getPartner() || state.getAccept1().getPartner() == 0
                            || Math.abs(state.getAccept1().getEnergy()) <500 ? 0 : state.getAccept1().getPartner() + 1;
                    int res2 = state.getDonor1().getPartner();
                    int res3 =
                            resid == state.getAccept2().getPartner()
                                    || state.getAccept2().getPartner() == 0
                                    || Math.abs(state.getAccept1().getEnergy()) <500 ? 0 : state.getAccept2().getPartner() + 1;
                    int res4 = state.getDonor2().getPartner();
                    System.out.println(state.printDSSPline(0));
                    System.out.println(state.getAccept1().getEnergy());
                    ArrayList<AminoAcid> l = new ArrayList<>();
                    if (aminoAcidSet.containsKey(res1)) {
                        l.add(aminoAcidSet.get(res1));
                    }
//                    if (aminoAcidSet.containsKey(res2)) {
//                        l.add(aminoAcidSet.get(res2));
//                    }
                    if (aminoAcidSet.containsKey(res3)) {
                        l.add(aminoAcidSet.get(res3));
                    }
//                    if (aminoAcidSet.containsKey(res4)) {
//                        l.add(aminoAcidSet.get(res4));
//                    }

                    hBondArrayListMap.put(aminoAcidSet.get(resid), l);

//                    String sss =  ?state.printDSSPline(0) + "\n" + state.getAccept1().getPartner() :"";
//                    System.out.println(" " + sss
//                    );
                }
            }


        } catch (StructureException e) {
            e.printStackTrace();
        }
    }


    public boolean process() {

        try {
            Chain chain = structure.getChainByPDB(proteinChain);// the first chain must be specific or prefer proteins with one chain

            // get all aminoacids
            List<Group> group = chain.getAtomGroups(GroupType.AMINOACID);
            for (Group g : group) {
                org.biojava.nbio.structure.AminoAcid amino = (org.biojava.nbio.structure.AminoAcid) g;
                AminoAcid aminoAcid = new AminoAcid();
                aminoAcid.setId(amino.getResidueNumber().getSeqNum());
                aminoAcid.setLabel(amino.getAminoType().toString());
//                aminoAcid.setProperty(amino.getProperties());
                aminoAcidSet.put(amino.getResidueNumber().getSeqNum(), aminoAcid);

            }
            setSecondaryStructure(group);


        } catch (StructureException e) {
            e.printStackTrace();
        }

        return false;
    }

    private void setStructure(String PBDID) {
        try {
//            AtomCache cache = new AtomCache();
//            cache.setCachePath("C:\\Users\\CECSAdmin\\Documents\\biojava");
            structure = StructureIO.getStructure(PBDID);
        } catch (IOException e) {
            e.printStackTrace();
        } catch (StructureException e) {
            System.out.println("Error message: " + e.getMessage());
        }
    }

    public Map<Integer, AminoAcid> getAminoAcidSet() {

        return aminoAcidSet;
    }

    public Map<AminoAcid, ArrayList<AminoAcid>> getBondArrayListMap() {
        return hBondArrayListMap;
    }
}

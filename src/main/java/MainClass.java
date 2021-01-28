import org.w3c.dom.Node;

import java.io.IOException;
import java.util.*;

public class MainClass {

    public static void main(String[] args) throws IOException {
        String directory = "exports/non-enzyme/";
        //4 each of oxidoreductases,transferase,hydroxylase
//        String[] pdbID = {"2naw"};
        //Conotoxins
//        String[] pdbID = {
//                "1a0m", "1akg", "1mxp", "1b45", "1e74", "1e75", "1e76", "2lr9", "1ieo", "1k64", "1mtq", "1hje", "1wct",
//                "1xga", "1y62", "1zlc", "2ajw", "2fqc", "2fr9", "2frb", "2gcz", "2ns3", "2ifi", "2ifj", "2ih7", "2b5p",
//                "2j15", "2juq", "2jur", "2jus", "2jut", "2ler", "2lz5", "2m3i", "2m62", "2md6", "2nay", "2naw", "3zkt",
//                "4ttl", "1ul2", "5t6v", "6efe", "1ag7", "1jlo", "1pqr", "1p1p", "1ttl", "1gib", "2jus", "1g1z", "1fu3",
//                "1f3k", "1eyo", "1omn", "1r9i", "1tt3", "1yz2", "2efz", "2km9", "2lmz", "2lo9", "2lxg", "2m61", "2mso",
//                "2msq", "2mw7", "2n7f", "2n8e", "2n8h", "2yen", "6bx9", "6ceg", "6cei", "2p4l"
//        };

        //Enzymes

        String[] enzyme = {
                "11AS", "1A26",
                "1A2J", "1A2P", "1A33", "1A49", "1A4M", "1A4S", "1A59", "1A5V", "1A5Z", "1A69", "1A77",
                "1A7U", "1A82", "1A8D", "1A8P", "1A8R", "1A95", "1A9U", "1A9X",
                "1AA6",
                "1VE9",
                "1ABO", "1AD4", "1ADE",
                "1ADO", "1AE1", "1AFW", "1AGJ", "1AI4", "1AJ5", "1AJA", "1AK2", "1AL8", "1ALN", "1AQ2", "1AQY", "1ARC",
                "1AUR", "1AUW", "1AV4", "1AW9", "1AY5", "1AYD", "1AYX", "1AZ3",
                "1WL9",
                "1B06", "1B0E", "1B0Z", "1B14",
                "1B1Y", "1B25", "1B31", "1B38", "1B49", "1B4U", "1B4V", "1B55", "1B57", "1B66", "1B6A", "1B6B", "1B6S",
                "1B74", "1B7Y", "1B80", "1B92", "1B9I",
                "1B9T",
                "1BA3", "1BAI",
                "1BC5",
                "1BD0", "1BD3", "1BDO",
                "4V40",
                "1BH5", "1BHJ", "1BIX", "1BK0", "1BK4", "1BLI", "1BMQ", "1BN6", "1BOX", "1BQ6", "1BS4", "1BT3", "1BT4",
                "1BU6",
                "1BUC", "1BVU", "1BW0", "1BWP", "1BYS", "1C02", "1C1D", "1C1H", "1C2P", "1C3F", "1C3R", "1C3U",
                "1C4A", "1C77", "1C7I", "1C7K", "1C7N", "1CCW", "1CD5", "1CEF", "1CEN",
                "1CF2",
                "1CG6", "1CIP", "1CJC",
                "1CKJ", "1CKN", "1CL0", "1CL2", "1CL6", "1CMV", "1CNS", "1CNZ", "1COM", "1CP2", "1CPM", "1CR0", "1CSK",
                "1CSM", "1CVL", "1CWR", "1CWU", "1CY0", "1CYX", "1D0Q", "1D3H", "1D4D", "1D6S", "1D6W", "1D7K", "1D8I",
                "1D8W", "1DAW", "1DCO", "1DCU", "1DD8", "1DDE", "1DEL", "1DFA", "1DHP", "1DHY", "1DIA", "1DIH", "1DIX",
                "1DJ9", "1DJN", "1DJO", "1DKM", "1DKU", "1DLG", "1DLJ", "1DLQ", "1DO8", "1DQA", "1DQI", "1DQP", "1DQX",
                "1DS0", "1DT1", "1DU4", "1DUC", "1DUG", "1DUP", "1DV1", "1DV7", "1DVG", "1DWK", "1DY3", "1DZT", "1E0T",
                "1E1O", "1E1R", "1E3D", "1E3P", "1E3U", "1E4L", "1E5E", "1E5L", "1E5S", "1E6B", "1E6P", "1E6U", "1E6Y",
                "1E7L", "1E7Y", "1E8Y", "1E92", "1EA1", "1EAF", "1ECJ", "1EDQ", "1EEX", "1EG9", "1EGH", "1EGU", "1EH7",
                "1EJ0", "1EJ2", "1EJB", "1EKK", "1EKP", "1EL5", "1ELU", "1EMV", "1ENI", "1EO2", "1EOI", "1EOM", "1EOV",
                "1EP2", "1EP9", "1EQC", "1EQJ", "1ES8", "1ESD", "1ESW", "1EUA",
                "1EUD", "1EUS", "1EUU", "1EVL", "1EVX", "1EX7", "1EYB", "1EYQ", "1EYY", "1EZ1", "1EZI", "1F07", "1F0R",
                "1F14", "1F1J", "1F28", "1F2V", "1F3L", "1F3P",
                "1F4L",
                "1F52", "1F5A", "1F5V", "1F6W", "1F75", "1F7T",
//                "1F83",
                "1F8A",
                "1F8I", "1F8Y", "1F9O", "1F9Z", "1FBT", "1FC7", "1FCB", "1FCQ", "1FF3", "1FFU", "1FG7", "1FH0",
                "1FHV",
                "1FK8", "1FO2", "1FO6", "1FO9",
                "6CIG",
                "1FR8", "1FS7", "1FW8", "1FWK", "1FWN", "1FWX", "1FX4", "1FXJ",
                "1FYE",
                "1G0C", "1G0H", "1G0R", "1G0W", "1G1A", "1G2O", "1G3K", "1G4E", "1G51", "1G58", "1G5T", "1G6L", "1G6T",
                "1G72",
                "1G93", "1G95", "1G98", "1G9Q", "1GA4", "1GA8", "1GEQ", "1GG1", "1GGV", "1GHR", "1GJW", "1GKX",
                "1XFG",
                "1GOF",
                "1GUP",
                "1H4V",
                "1H6J", "1H7X", "1HC7", "1HD7", "1HDR", "1HE2", "1HFC", "1HHS", "1HI3", "1HMY", "1HND",
                "1HNU", "1HO4", "1HP5", "1HPL", "1HTP", "1HUC", "1HW6", "1HX3", "1HXD", "1I0S", "1I12",
                "1I1Q",
                "1I44",
                "1I59", "1I6P", "1I72", "1I8A", "1I9T", "1I9Z", "1IA1", "1IAH", "1IG8", "1IH7", "1II0", "1IIP", "1ILZ",
                "1IMA", "1INC", "1INJ", "1INO", "1IO2", "1IO7", "1IOF", "1IOW", "1IRB", "1ISA", "1ISO", "1IUS", "1IVP",
                "1J71", "1J7L", "1J80", "1J93", "1J97", "1J9M", "1J9Q", "1JA1", "1JA9", "1JAE", "1JAN", "1JB9", "1JBB",
                "1JBP",
                "1JBV", "1JC4", "1JD0", "1JD3", "1JDA", "1JDF", "1JDR", "1JDX", "1JEH", "1JEJ", "1JF9", "1JFV", "1JGM",
                "1JH8", "1JHE", "1JIN", "1JK7", "1JKH", "1JLN", "1JMF", "1JN3", "1JNW", "1JOL", "1JPR", "1JQ5", "1JSV",
                "1JUK", "1JWO", "1JXZ", "1K06", "1K2O", "1K89", "1KAA",
                "1KAP",
                "1KAX", "1KDN", "1KEV", "1KI2", "1KLT",
                "1KOQ",
                "1KVA", "1KVW", "1L97", "1LAM", "1LCJ", "1LDG", "1LLO", "1LOP", "1LSG", "1LSP", "1MAC", "1MAR", "1MDL",
                "1MHY",
                "1MKA", "1MLD", "1MOQ", "1MPP", "1MUY", "1NAS", "1NEC", "1NGS", "1NHP", "1NHT", "1NIR", "1NMT",
                "1NNA", "1NOS", "1NOZ", "1NSA", "1OAC", "1OAT", "1OHJ", "1OIL", "1ONR", "1OPR", "1ORB", "1OTH", "1OYB",
                "1PBK",
                "1PGN", "1PGS", "1PHK", "1PHM", "1PHP", "1PHR", "1PI2", "1PJC", "1PLU", "1PMK", "1PML", "1PMT", "1POO",
                "1POX", "1PTR", "1PUD", "1PVD", "1QA7", "1QAE", "1QAM", "1QAS", "1QB8", "1QBB", "1QBG", "1QBI", "1QBQ",
                "1QCF", "1QCO", "1QCX", "1QCZ", "1QDQ", "1QDR", "1QF7", "1QFS", "1QGQ", "1QH5", "1QH7", "1QHA", "1QHH",
                "1QHX", "1QID", "1QJ5", "1QJC", "1QJI", "1QK3", "1QLB", "1QLT", "1QM6", "1QMG", "1QMH", "1QMV", "1QNF",
                "1QPC", "1QPO", "1QRE", "1QRF", "1QRK", "1QSA", "1QTR", "1QVB", "1R2F", "1RK2", "1RLR", "1RLW", "1RXF",
                "1SET", "1SMR", "1SVP", "1TF4", "1TKI",
                "1TYB",
                "1UBV", "1VIF", "1VNC", "1VPE",
                "1WWB",
                "1XAA", "1XAN",
                "1XGS", "1XIB", "1XIK", "1XJO", "1XKJ", "1XNC", "1XO1", "1XPB", "1XPS", "1XSO", "1XYS", "1XZA", "1YAL",
                "1YDV", "1YFO", "1YGE", "1YGH", "1YLV", "1YME", "1YPN", "1YPP", "1YTN", "1ZIN", "1ZRM", "1ZYM", "2A0B",
                "2AHJ",
                "2APS", "2AXE", "2BSP",
                "2CI2",
                "2CND", "2DAP", "2DHN", "2DIK", "2DOR", "2DUB", "2EQL",
                "2ER6",
                "2ERK",
                "2EST",
                "2EUG", "2F3G", "2FGI", "2FHE", "2FIV", "2FKE", "2FOK", "2FUA", "2FUS", "2GAC", "2GAR",
                "2GD1",
                "2GEP",
                "2GLT", "2GSQ", "2HAD", "2HPD", "2HPR",
                "2MAD",
                "2MAN", "2MAS", "2MBR", "2MIN", "2MJP", "2NAD", "2NSY",
                "2PF1",
                "2PKA", "2POL", "2PVA", "2QR2", "2RSL", "2RUS", "2SQC", "2SRC", "2TDT", "2THF", "2TLI", "2TMK", "2TOH",
                "2TPL", "2TS1", "2TSC", "2UAG", "2UBP", "2UDP", "2UKD", "2USH",
                "2VP3",
                "3BC2", "3BIF", "3BIR", "3BLM",
                "3BLS", "3BTO", "3CBH", "3CD2", "3CEV", "3CGT", "3CLA", "3CMS", "3CSC", "3CSU", "3CYH", "3DAA", "3DHE",
                "3DMR", "3ECA", "3ENG", "3GCB", "3HAD", "3PBH", "3PMG", "3PNP",
                "3PRK",
                "3PTD", "3RAN",
                "3RUB",
                "3SIL",
                "3SLI", "3STD", "3TGL",
                "3THI", "3VGC",
                "4AIG",
                "4DCG", "4FBP", "4GSA", "4LZM", "4MAT", "4MDH", "4NOS", "4OTB", "4PAH", "4PBG", "4PFK", "4PGM", "4TMK",
                "5EAU", "5KTQ", "5YAS", "6ENL", "6GSS", "6TAA", "7AAT", "7ACN", "7ATJ", "7CAT", "7REQ", "8A3H", "8CHO",
                "8LPR", "9GAC"
        };

//non enzymes
        String[] nonenzyme = {
                "1A04", "1A0K",
                "1A0S",
                "1A1R", "1A1X", "1A21", "1A2B", "1A3Z", "1A44", "1A45", "1A4X", "1A62", "1A64",
                "1A7G",
                "1A7W", "1A8A", "1A8O", "1A92", "1A99", "1AAZ", "1AB1", "1AHQ", "1AIE", "1AIL", "1AJJ", "1ALU",
                "1AOH", "1AOX", "1AQB", "1AQD", "1AQE", "1AR0", "1AS0", "1AS4", "1ATG", "1AUE", "1AUN", "1AUV", "1AVU",
                "1AWP", "1AXI",
                "1AY1",
                "1AYF", "1AYI",
                "1AYM",
                "1B09", "1B0L", "1B0O", "1B0Y", "1B1U", "1B3A", "1B63",
                "1B67", "1B71", "1B7D", "1B7V", "1B88", "1B8Z", "1B9N", "1B9W", "1BA2", "1BAS", "1BD8", "1BFE", "1BFT",
                "1BGE", "1BH2", "1BHD", "1BKB", "1BM0",
                "1BM3",
                "1BMB", "1BMG", "1BOY", "1BRX", "1BTG", "1BTN", "1BV1",
                "1BX8", "1BXM", "1BXT", "1BY7", "1BYF", "1C1L", "1C48", "1C4P", "1C5E", "1C5K", "1C6O", "1C94", "1CAG",
                "1CAU", "1CC7", "1CDH", "1CDM", "1CDT", "1CFM", "1CFW", "1CI4", "1CNO", "1COT", "1CPC", "1CQ4", "1CS3",
                "1CSP", "1CT5", "1CXA", "1CYO", "1CZD", "1CZQ", "1D00", "1D06", "1D2E", "1D5T", "1D5W",
                "1D7P",
                "1DDV",
                "1DFN", "1DJ8", "1DK8", "1DOK", "1DOT", "1DQO", "1DQZ", "1DTJ", "1DUW", "1DV8", "1DVN", "1DZI", "1E00",
                "1E0B", "1E29", "1E2U", "1E4J", "1E7C",
                "1GK6",
                "1E7Z", "1E87", "1E9L", "1E9M", "1EA3", "1EAJ",
                "1ED1", "1EE4", "1EFT", "1EG4", "1EGI", "1EHH", "1EI7", "1EJ4", "1EKG", "1EKS", "1EPB", "1EPU", "1ERN",
                "1ET6", "1ET9", "1EWF", "1EYH", "1EZG", "1F0M", "1F2L",
                "1F2X",
                "1F47", "1F56", "1F5M", "1F5N", "1F7C",
                "1FAO", "1FBQ", "1FCY", "1FD3", "1FH2", "1FHG", "1FHW", "1FL1", "1FLM", "1FNA", "1FR9", "1FSO", "1FT5",
                "1FTJ", "1FVU", "1FW4", "1FXD", "1FYH", "1G1C", "1G33", "1G3J", "1G43", "1G4R", "1G5I", "1G5Y", "1G62",
                "1G6H", "1G6N", "1G7C", "1G7S", "1G8I", "1G9O", "1GE8", "1GPC", "1GZI", "1H4Y", "1H75", "1H8N", "1H9G",
                "1H9K", "1HDF", "1HFA", "1HG4", "1HH0", "1HH5", "1HH8", "1HJP", "1HOE", "1HQ3", "1HQN", "1HQO",
                "1HTJ",
                "1HUS", "1HXI", "1HZ5", "1HZG", "1I07", "1I31", "1I4U", "1I6A", "1I7E", "1I81", "1I92",
                "1M8Z",
                "1IC0",
                "1IC2", "1IE8", "1IFG", "1IIT", "1IKE",
                "1IKF",
                "1IL1",
                "1ILR",
                "1ILS", "1IN5",
                "1IND",
                "1INN", "1INR",
                "1IO3", "1ION", "1IOP", "1IOZ", "1IRD", "1IRN", "1IUZ", "1IXG", "1J6Z", "1J73", "1J7A", "1J8Q", "1J8S",
                "1J8Y",
                "1JAF", "1JAH", "1JB3", "1JB6", "1JBC", "1JCF", "1JD1", "1JDO", "1JET", "1JF0", "1JGJ", "1JI6",
                "1JJH", "1JL5", "1JLJ", "1JLM", "1JLY", "1JMW", "1JOB", "1JOT", "1JQF", "1JRR", "1JVX", "1JW8",
                "1JY3",
                "1K0K", "1K33", "1KDJ", "1KIV", "1KLO",
                "1KMB",
                "1KOE", "1KVE",
                "1LAF",
                "1LE4", "1LEN", "1LFO", "1LGH",
                "1LIN", "1LIT", "1LKF", "1LLA", "1LOU",
                "1LT5",
                "1LVE", "1LVK", "1MDT", "1MFF", "1MH1", "1MHO", "1MJK",
                "1MOF", "1MOL", "1MPC", "1MRG", "1MRP", "1MSA", "1MSC", "1MUP", "1MYK", "1MYT", "1MZM", "1NAT", "1NCH",
                "1NCO", "1NCX", "1NDD", "1NFO", "1NFP", "1NFT", "1NG1", "1NK1", "1NKD", "1NPS", "1NSF", "1NT3", "1NTN",
                "1OPC", "1OR3", "1ORC", "1OVB", "1OXY", "1PCZ", "1PGB", "1PHN", "1PLF", "1PSZ", "1PTX", "1QAD",
                "1QAW",
                "1QDE", "1QDN", "1QDV", "1QE6", "1QFG", "1QFV", "1QG7", "1QGH", "1QHV", "1QJ9", "1QJA", "1QJP", "1QK8",
                "1QKM", "1QKX", "1QLP", "1QM7", "1QME", "1QOK",
                "1QOV",
                "1QQ1", "1QQF", "1QSC", "1QSU", "1QTO", "1QTP",
                "1QVC", "1R69", "1RCB", "1RH4",
                "1WU3",
                "1ROP", "1RPJ", "1RRG",
                "1P30",
                "1SCE", "1SFP", "1SKZ", "1SWG",
                "1UP1", "1VFY", "1VIN", "1VPN", "1WHO", "1XCA",
                "1XXA",
                "1YFP",
                "1YHB",
                "1YPA",
                "1YTT", "1ZEI",
                "1ZFP",
                "1ZOP", "2A2U", "2ARC", "2BNH", "2CAV", "2CBL", "2ERA", "2ERL", "2FAL",
                "2FB4",
                "2FCR", "2FD2", "2FDN",
                "2FGF", "2FHA", "2FIB", "2FIT", "2GAL", "2GDM", "2GWX", "2HEX", "2HIP", "2HMQ", "2IGD", "2IMM", "2IMN",
                "2INT", "2IZA", "2LIG", "2LIS", "2LIV", "2OMF", "2PLH", "2PSP", "2TCT", "2TDX", "2TEP", "2TGI", "2TIR",
                "2TMY", "2TN4", "2TNF", "2TRH", "2TSS", "2UGI", "2UTG", "2VPF", "2WBC",
                "2OZ9",
                "2YGS", "3CAR", "3CLN",
                "3CYR", "3EIP", "3ERD", "3FIS", "3GBP", "3KAR", "3LBD", "3OVO", "3POR", "3PRN", "3PSR", "3PYP", "3RHN",
                "3SEB", "3SEM", "3SSI", "3VUB", "451C", "4BCL", "4BJL", "4ICB", "4LVE", "4MON", "4OVO", "4PAL", "5PTI",
                "7ABP", "7AME", "7PAZ", "7PCY", "7PTI", "7RXN", "9WGA"
        };

//        for (String id : nonenzyme) {
//
//
//            Preprocess preprocess = new Preprocess(id, "A");
//            Map<Integer, AminoAcid> aminoAcids = preprocess.getAminoAcidSet();//
//            Map<AminoAcid, ArrayList<AminoAcid>> aminoAcidsHBonds = preprocess.getBondArrayListMap();//
//
////        System.out.println(preprocess.printDSSPFormat());
//            ProteinGraph proteinGraph = new ProteinGraph(Bond.class);
//            proteinGraph.addAAwithPeptideBonds(aminoAcids);////
//            proteinGraph.addHydrogenBonds(aminoAcidsHBonds);//
//
////        System.out.println(proteinGraph.findMaximumCliques());
////        proteinGraph.visualize("Visualizer");
////        proteinGraph.exportGraph(ProteinGraph.EXPORT_TYPE_DOT, directory + id);
//            proteinGraph.exportGraph(ProteinGraph.EXPORT_TYPE_CSV, directory + id);
////        proteinGraph.exportGraph(ProteinGraph.EXPORT_TYPE_MATRIX, directory + pdbID[0]);
//        }
        String[] dirs = {
                "C:\\Users\\CECSAdmin\\OneDrive - University of Tennessee at Chattanooga" +
                "\\Projects\\ProteinGraph\\exports\\non-enzyme\\",
                "C:\\Users\\CECSAdmin\\OneDrive - University of Tennessee at Chattanooga" +
                "\\Projects\\ProteinGraph\\exports\\enzyme\\"};

        String[][] filenames={nonenzyme,enzyme};
        try {
            new Classifier().saveInput(dirs,filenames);
//            new Classifier().binaryClassification();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

}
import org.w3c.dom.Node;

import java.io.IOException;
import java.util.*;

public class MainClass {

    public static void main(String[] args) throws IOException {
        String directory = "exports/";
        //4 each of oxidoreductases,transferase,hydroxylase
//        String[] pdbID = {"1mxp"};
        //Conotoxins
        String[] pdbID = {"1a0m", "1akg", "1mxp", "1b45", "1e74", "1e75", "1e76", "2lr9", "1ieo", "1k64", "1mtq", "1hje",
                "1wct", "1xga", "1y62", "1zlc", "2ajw", "2fqc", "2fr9", "2frb", "2gcz", "2ns3", "2ifi", "2ifj", "2ih7",
                "2b5p", "2j15", "2juq", "2jur", "2jus", "2jut", "2ler", "2lz5", "2m3i", "2m62",

                "2md6", "2nay", "2naw", "3zkt", "4ttl", "1ul2", "5t6v", "6efe", "1ag7", "1jlo",
                "1pqr", "1p1p", "1ttl", "1gib", "2jus", "1g1z", "1fu3", "1f3k", "1eyo", "1omn",

                "1r9i", "1tt3", "1yz2", "2efz", "2km9", "2lmz", "2lo9", "2lxg", "2m61", "2mso", "2msq", "2mw7", "2n7f", "2n8e",
                "2n8h", "2yen", "6bx9", "6ceg", "6cei", "2p4l"};
        for (String id : pdbID) {


            Preprocess preprocess = new Preprocess(id, "A");
            Map<Integer, AminoAcid> aminoAcids = preprocess.getAminoAcidSet();//
            Map<AminoAcid, ArrayList<AminoAcid>> aminoAcidsHBonds = preprocess.getBondArrayListMap();//

//        System.out.println(preprocess.printDSSPFormat());
            ProteinGraph proteinGraph = new ProteinGraph(Bond.class);
            proteinGraph.addAAwithPeptideBonds(aminoAcids);////
            proteinGraph.addHydrogenBonds(aminoAcidsHBonds);//

//        System.out.println(proteinGraph.findMaximumCliques());
//        proteinGraph.visualize("Visualizer");
//        proteinGraph.exportGraph(ProteinGraph.EXPORT_TYPE_DOT, directory + pdbID[0]);
            proteinGraph.exportGraph(ProteinGraph.EXPORT_TYPE_CSV, directory + id);//
//        proteinGraph.exportGraph(ProteinGraph.EXPORT_TYPE_MATRIX, directory + pdbID[0]);
        }
        try {
            new Classifier(pdbID).startClustering();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

//    static class Node {
//
//        int val;
//        boolean visited;
//        Node left, right;
//
//        Node(int d) {
//            val = d;
//            left = right = null;
//            visited = false;
//        }
//    }

//    static class BinaryTree {
//
//        Node head;
//
//        /* Given a binary search tree and a number,
//        inserts a new node with the given number in
//        the correct place in the tree. Returns the new
//        root pointer which the caller should then use
//        (the standard trick to avoid using reference
//        parameters). */
//        Node insert(Node node, int data) {
//
//		/* 1. If the tree is empty, return a new,
//		single node */
//            if (node == null) {
//                return (new Node(data));
//            } else {
//
//                /* 2. Otherwise, recur down the tree */
//                if (data <= node.val) {
//                    node.left = insert(node.left, data);
//                } else {
//                    node.right = insert(node.right, data);
//                }
//
//                /* return the (unchanged) node pointer */
//                return node;
//            }
//        }
//    }

    /* Given a non-empty binary search tree,
    return the minimum data value found in that
    tree. Note that the entire tree does not need
    to be searched. */
//    int minvalue(Node node) {
//        Node current = node;
//
//        /* loop down to find the leftmost leaf */
//        //while (current.left != null) {
//        //current = current.left;
//        //}
//        //return (current.data);
//        HashSet<Integer> s = new HashSet<>();
//        inorder(node, 2, s);
//        List<Integer> mainList = new ArrayList<>();
//        mainList.addAll(s);
//        return mainList.size() > 1 ? mainList.get(1) : -1;
//    }

//    static void inorder(Node root, int c, HashSet s) {
//        if (root != null && s.size() < c) {
//            inorder(root.left, c, s);
//            if (s.size() < c) {
//                if (!s.contains(root.val)) s.add(root.val);
//                inorder(root.right, c, s);
//            }
//        }
//
//    }
//
//    static boolean isAVLTree(Node root) {
//        if (root == null) return false;
//
//        int bf = height(root.right) - height(root.left);
//        System.out.println(root.val);
//        if (bf == -1 || bf == 0) {
//            return true;
//        }
//        return isAVLTree(root.left) && isAVLTree(root.right);
//
//
//    }
//
//    static int height(Node root) {
//        if (root == null) return 0;
//        System.out.println(root.val);
//        return Math.max(height(root.left), height(root.right)) + 1;
//    }
//
//    /*ignore these*/
//    public static long foo(int n) {
//        if (n <= 1) return 0;
//        else if (n == 2) return 1;
//        else if (n == 3) return 2;
//
//        long res0 = 0;
//        long res1 = 0;
//        long res2 = 1;
//        long res3 = 2;
//        for (int i = 3; i < n; i++) {
//            long res4 = 4 * res3 + 2 * res2 + 3 * res1 + res0;
//            res0 = res1;
//            res1 = res2;
//            res2 = res3;
//            res3 = res4;
//
//        }
//        return res3;
//    }
}

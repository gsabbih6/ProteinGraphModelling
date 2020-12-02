import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.clustering.algorithm.Distance;
import org.deeplearning4j.clustering.cluster.Cluster;
import org.deeplearning4j.clustering.cluster.ClusterSet;
import org.deeplearning4j.clustering.cluster.Point;
import org.deeplearning4j.clustering.kmeans.KMeansClustering;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.DropoutLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.plot.BarnesHutTsne;
import org.deeplearning4j.plot.Tsne;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.AdaDelta;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import smile.clustering.GMeans;
import smile.clustering.KMeans;
import smile.manifold.TSNE;
import smile.plot.swing.Canvas;
import smile.plot.swing.Palette;
import smile.plot.swing.*;

import javax.swing.*;
import java.awt.*;
import java.io.File;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Classifier {
    private long seed = 10123;
    private String[] files;
// load and preprocess files from directory
    // Build Deep Network with a bias and an output vector based on the number of nodes
    // save output as in a list of INDArrays
    // Use the the list as input to G/Kmeans clustering
    // Reduce the dimension to 3 and plot using t-SNE

    public Classifier(String[] pdbID) {
        initialize(pdbID);
    }

    private void initialize(String[] pdbID) {
        this.files = pdbID;
    }

    public void startClustering() throws IOException, InterruptedException {
        List<TSNE> l = new ArrayList<>();
        List<INDArray> l1 = new ArrayList<>();
        INDArray Ge;
        for (int i = 1; i < files.length; i++) {

        }
        for (String file : files) {
            Ge = getGraphEmbedding(file);
//            l1.add(Ge);
            TSNE tsne = Gmeanstsne(Ge);
            l.add(tsne);

        }
//        Ge=Nd4j.concat(0, l1.toArray(new INDArray[l1.size()]));
//        TSNE tsne = Kmeanstsne(Ge);
//        l.add(tsne);
        plot(l);
    }

    public TSNE Kmeanstsne(INDArray Ge) {
        System.out.println(Ge);
        System.out.println(Ge.columns());
        KMeans cluster = KMeans.fit(Ge.toDoubleMatrix(), 4);
        TSNE tsne = tsne(cluster.centroids);
        return tsne;
    }

    private TSNE tsne(double[][] centroids) {
        return new TSNE(centroids, 2, 50, 200, 1000);
    }

    public TSNE Gmeanstsne(INDArray Ge) {
        System.out.println(Ge);
        System.out.println(Ge.columns());
        GMeans cluster = GMeans.fit(Ge.toDoubleMatrix(), 4);
        TSNE tsne = tsne(cluster.centroids);
        return tsne;

    }

    public void plot(List<TSNE> tsne) {
        Canvas canvas = ScatterPlot.of(tsne.get(0).coordinates, '#', new Color((int) (Math.random() * 0x1000000))).canvas();
        for (int i = 1; i < tsne.size(); i++) {
            canvas.add(ScatterPlot.of(tsne.get(i).coordinates, '#', new Color((int) (Math.random() * 0x1000000))));
        }
        try {
            canvas.window();
        } catch (InvocationTargetException | InterruptedException e) {
            e.printStackTrace();
        }
    }

    public INDArray getGraphEmbedding(String filename) throws IOException, InterruptedException {
        DataSet dataset =
                readData("C:\\Users\\CECSAdmin" +
                        "\\OneDrive - University of Tennessee at Chattanooga" +
                        "\\Projects\\ProteinGraph\\exports\\" + filename + ".csv");
        List<DataSet> set = dataset.asList();
//
        for (int i = 0; i < set.size(); i++) {
            System.out.println("input");
            System.out.println(set.get(i).getLabels());
            INDArray Hv = Nd4j.zeros(set.get(i).numInputs());

            // message passing phase
            for (int j = 0; j < set.size(); j++) { // neighbor
                if (j != i) {
                    INDArray message = buildMessagePassingModel(set.get(j), set.size());
                    Hv = Hv.add(message);
                }
            }

            // Update Phase
            Hv.add(set.get(i).getFeatures());
            DataSet set1 = new DataSet(Hv, Hv);
            INDArray update = buildUpdatePhaseModel(set1, set.size());
            set.get(i).setFeatures(update);
            System.out.println("output");
            System.out.println(set.get(i).getFeatures());

        }
        //Graph Embedding
        INDArray Ge = Nd4j.zeros(set.get(0).numInputs());
        for (DataSet s : set) {
            Ge = Ge.add(s.getFeatures());
        }
        return Ge;
    }


    private INDArray buildMessagePassingModel(DataSet set, int numNodes) { //mxn

        int featureSize = set.numInputs();
//        System.out.println(featureSize);
        MultiLayerConfiguration model = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new AdaDelta()) // try other updaters later
                .list()
                .layer(new DenseLayer.Builder().nIn(featureSize).nOut(featureSize)
                        .activation(Activation.RELU)
                        .build())
//                .layer(new DropoutLayer.Builder().dropOut(0.5).nIn(featureSize).nOut(featureSize).build())// Just for test
                .layer(new DenseLayer.Builder().nIn(featureSize).nOut(featureSize)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation(Activation.SOFTMAX)
                        .name("last")
                        .nIn(featureSize).nOut(featureSize).build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(model);
        net.init();
//        net.setEpochCount(5000);
//        DataSet s = new DataSet();
//        s.setFeatures(set.getFeatures());
        net.setEpochCount(1);
        net.fit(set);
//        net.getOutputLayer().params();
        return net.output(set.getFeatures());
    }

    private INDArray buildUpdatePhaseModel(DataSet set, int numNodes) { //mxn
        int featureSize = set.numInputs();
        MultiLayerConfiguration model = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .updater(new AdaDelta()) // try other updaters later
                .list()
                .layer(new DenseLayer.Builder().nIn(featureSize).nOut(featureSize)
                        .activation(Activation.RELU)
                        .build())
//                .layer(new DropoutLayer.Builder().dropOut(0.5).nIn(featureSize).nOut(featureSize).build())// Just for test
//                .layer(new DenseLayer.Builder().nIn(numNodes).nOut(numNodes)
//                        .activation(Activation.RELU)
//                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation(Activation.SOFTMAX)
                        .nIn(featureSize).nOut(featureSize).build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(model);
        net.init();
//        net.setEpochCount(5000);
        DataSet s = new DataSet();
//        s.setFeatures(set);
        net.setEpochCount(1);
        net.fit(set);
        return net.output(set.getFeatures());
    }

    private DataSet readData(String directory) throws IOException, InterruptedException {
        int batchSize = 1000;
        RecordReader rr = new CSVRecordReader();
        NormalizerStandardize normaliser = new NormalizerStandardize();
        rr.initialize(new FileSplit(new File(directory)));

        DataSetIterator iter = new RecordReaderDataSetIterator.Builder(rr, batchSize).build();
        normaliser.fit(iter);
        iter.setPreProcessor(normaliser);
//        new MapFileSequenceRecordReader();
        return iter.next();
    }

    private void computeEmbeddingPerGraph() {

    }

    private void updateAllEmbedding() {

    }

    private void clusterWithKmeans(int clusterCount, int maxIterationCount, List<INDArray> vectors) throws IOException {
        //1. create a kmeanscluster instance
        String distanceFunction = "cosinesimilarity";
        KMeansClustering kmc = KMeansClustering.setup(clusterCount, maxIterationCount, Distance.EUCLIDEAN,
                true);
//        new KMeansClustering(clusterCount,maxIterationCount,distanceFunction);
        System.out.println(vectors.get(0));
//        GMeans gmeans = new GMeans();
        ClusterSet cs = kmc.applyTo(Point.toPoints(vectors));
        BarnesHutTsne tsne = new BarnesHutTsne.Builder()
                .setMaxIter(500).theta(0.5)
                .normalize(false)
                .learningRate(500)
                .useAdaGrad(false)
//                .usePca(false)
                .perplexity(1)
                .build();

        tsne.plot(vectors.get(0),
                4, Arrays.asList("s", "s", "s", "s", "s", "s", "s", "s", "s", "s", "s", "s", "s", "s"),
                "C:\\Users\\CECSAdmin\\OneDrive - University of Tennessee at Chattanooga\\Projects\\ProteinGraph\\exports\\tes.csv");

//        plotdata(cs.getClusters());

    }

    private void plotdata(double[][] clusters) {
        XYSeriesCollection c = new XYSeriesCollection();


        int dscounter = 1; //use to name the dataseries
        XYSeries series = new XYSeries("S" + dscounter);
        for (double[] ds : clusters) {
//            INDArray features = ds.getCenter().getArray();
//            INDArray outputs = ds.getCenter().getArray();

            int nRows = ds.length;

            for (int i = 0; i < nRows; i++) {
                series.add(ds[i], ds[i]);
            }

            c.addSeries(series);
        }

        String title = "title";
        String xAxisLabel = "xAxisLabel";
        String yAxisLabel = "yAxisLabel";
        PlotOrientation orientation = PlotOrientation.VERTICAL;
        boolean legend = false;
        boolean tooltips = false;
        boolean urls = false;
        //noinspection ConstantConditions
        JFreeChart chart = ChartFactory.createScatterPlot(title, xAxisLabel, yAxisLabel, c, orientation, legend, tooltips, urls);
        JPanel panel = new ChartPanel(chart);

        JFrame f = new JFrame();
        f.add(panel);
        f.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
//        f.pack();
        f.setTitle("Training Data");

        f.setVisible(true);

    }

    private void plotdata(ArrayList<DataSet> DataSetList) {
        XYSeriesCollection c = new XYSeriesCollection();


        int dscounter = 1; //use to name the dataseries
        XYSeries series = new XYSeries("S" + dscounter);
        for (DataSet ds : DataSetList) {
            INDArray features = ds.getFeatures();
            INDArray outputs = ds.getLabels();

            int nRows = features.rows();

//            for (int i = 0; i < nRows; i++) {
            series.add(features.getDouble(), outputs.getDouble());
//            }


        }
        c.addSeries(series);
        String title = "title";
        String xAxisLabel = "xAxisLabel";
        String yAxisLabel = "yAxisLabel";
        PlotOrientation orientation = PlotOrientation.VERTICAL;
        boolean legend = false;
        boolean tooltips = false;
        boolean urls = false;
        //noinspection ConstantConditions
        JFreeChart chart = ChartFactory.createScatterPlot(title, xAxisLabel, yAxisLabel, c, orientation, legend, tooltips, urls);
        JPanel panel = new ChartPanel(chart);

        JFrame f = new JFrame();
        f.add(panel);
        f.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
//        f.pack();
        f.setTitle("Training Data");

        f.setVisible(true);
    }
}
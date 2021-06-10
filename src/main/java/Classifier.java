
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.io.labels.PathLabelGenerator;
import org.datavec.api.io.labels.PathMultiLabelGenerator;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.FileRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.writable.Writable;
import org.deeplearning4j.clustering.algorithm.Distance;
import org.deeplearning4j.clustering.cluster.ClusterSet;
import org.deeplearning4j.clustering.cluster.Point;
import org.deeplearning4j.clustering.kmeans.KMeansClustering;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIteratorSplitter;
import org.deeplearning4j.datasets.iterator.FileSplitDataSetIterator;
import org.deeplearning4j.datasets.iterator.INDArrayDataSetIterator;
import org.deeplearning4j.datasets.iterator.callbacks.DataSetDeserializer;
import org.deeplearning4j.datasets.iterator.file.FileDataSetIterator;
import org.deeplearning4j.datasets.iterator.parallel.FileSplitParallelDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.plot.BarnesHutTsne;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.jetbrains.annotations.NotNull;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.general.Dataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.nd4j.common.primitives.Pair;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.*;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import smile.clustering.GMeans;
import smile.clustering.KMeans;
import smile.manifold.TSNE;
import smile.plot.swing.Canvas;
import smile.plot.swing.*;

import javax.swing.*;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.net.URI;
import java.net.URISyntaxException;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import java.util.stream.StreamSupport;

public class Classifier {
    private long seed = 10123;
    private String[] files;
    private int EbeddingNum = 2050;
    private int iteration = 1;

    private List<INDArray> generateEmbedding(String dir, String[] files) throws IOException, InterruptedException {

        List<INDArray> l1 = new ArrayList<>();
        INDArray Ge;
        AtomicInteger count = new AtomicInteger();
        List<String> fileList = Arrays.asList(files);

        fileList.parallelStream().forEach((file) -> {
            try {
                l1.add(getGraphEmbeddingParallel(dir, file));
            } catch (IOException e) {
                e.printStackTrace();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            System.out.println("processed file " + count.getAndIncrement() + "/" + files.length);
        });


//        for (String file : files) {
//            Ge = getGraphEmbeddingParallel(dir, file);
//            l1.add(Ge);
//            System.out.println("processed file " + count.getAndIncrement() + "/" + files.length);
//        }
        return l1;
    }

    public void saveInput(String[] dir, String[][] list) throws IOException, InterruptedException {
        int count = 0;
        String l = dir[0];

        String[] h = list[0];
        List<INDArray> l1 = generateEmbedding(dir[0], list[0]);//enzymes
//        INDArray labelsNonEnzymes = Nd4j.zeros(l1.size(), 1);
        INDArray labelsNonEnzymes = Nd4j.concat(1, Nd4j.ones(l1.size(), 1), Nd4j.zeros(l1.size(), 1));
        System.out.println(labelsNonEnzymes);
        List<INDArray> l2 = generateEmbedding(dir[1], list[1]);// non-enzymes
        INDArray labelsEnzymes = Nd4j.concat(1, Nd4j.zeros(l2.size(), 1), Nd4j.ones(l2.size(), 1));
        System.out.println(labelsEnzymes);
        l1.addAll(l2);

        INDArray features = Nd4j.concat(0, l1.toArray(new INDArray[l1.size()]));
        System.out.println(features);
        INDArray labels = Nd4j.concat(0, labelsNonEnzymes, labelsEnzymes);
        System.out.println("la " + labels);

        DataSet set = new DataSet(features, labels);
        set.save(new File("C:\\Users\\CECSAdmin\\OneDrive - University of Tennessee at Chattanooga" +
                "\\Projects\\ProteinGraph\\Preprocessed\\data"));

    }

    public void binaryClassification() throws IOException, InterruptedException {

        //Start binary classification
//        FileDataSetIterator iter = new FileDataSetIterator(
//                new File("C:\\Users\\CECSAdmin\\OneDrive - University of Tennessee at Chattanooga" +
//                        "\\Projects\\ProteinGraph\\exports"), false, new Random(), 100, null);

        DataSet alldataset = new DataSet();
        alldataset.load(new File("C:\\Users\\CECSAdmin\\OneDrive - University of Tennessee at Chattanooga" +
                "\\Projects\\ProteinGraph\\Preprocessed\\data"));
        alldataset.shuffle();
        alldataset.normalize();
        SplitTestAndTrain split = alldataset.splitTestAndTrain(0.95);
        DataSet train = split.getTrain();
        DataSet test = split.getTest();
        System.out.println(test);
        //Initialize the user interface backend
        UIServer uiServer = UIServer.getInstance();
        //Configure where the network information (gradients, score vs. time etc) is to be stored. Here: store in memory.
        StatsStorage statsStorage = new InMemoryStatsStorage();         //Alternative: new FileStatsStorage(File), for saving and loading later
        //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
        uiServer.attach(statsStorage);
        long nRows = train.numInputs();
        long nCols = train.getFeatures().columns();
        System.out.println("nRows= " + nRows + "nCols= " + nCols);
        MultiLayerNetwork model = model();//(nRows, nCols);
        model.setListeners(new StatsListener(statsStorage));
        model.init();
        model.setEpochCount(1000);
        model.fit(train);

//        System.out.println("\n**************** Example finished ********************");
//        model.save(new File("C:\\Users\\CECSAdmin\\OneDrive - University of Tennessee at Chattanooga" +
//                "\\Projects\\ProteinGraph\\exports\\model_150"));
//
//evaluate the model_30 on the test set
        Evaluation eval = new Evaluation(2);
        INDArray output = model.output(test.getFeatures());
        eval.eval(test.getLabels(), output);
//        log.info(eval.stats());

        System.out.println(eval.stats());
    }

    public MultiLayerNetwork model() {
        int seed = 123;
        double learningRate = 0.5;

        int numInputs = EbeddingNum;
        int numOutputs = 2;
        int numHiddenNodes = 1000;
        int numHiddenNodes1 = 512;
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .weightInit(WeightInit.XAVIER)
//                .op
//                .updater(new AdaGrad(learningRate))
                .updater(new Nesterovs(learningRate, 0.5))
                .list()
                .layer(new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                        .activation(Activation.RELU)
                        .build())
                .layer(new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes1)
                        .activation(Activation.RELU)
                        .build())
//                .layer(new DenseLayer.Builder().nIn(numHiddenNodes1).nOut(numHiddenNodes1)
//                        .activation(Activation.RELU)
//                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.SOFTMAX)
                        .nIn(numHiddenNodes1).nOut(numOutputs).build())
                .build();
        //Then add the StatsListener to collect this information from the network, as it trains
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        return model;
    }

    public MultiLayerNetwork mpnn(long nRows, long nCol) {
        int seed = 123;
        double learningRate = 0.5;
        int batchSize = 50;
        int nEpochs = 100;
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .weightInit(WeightInit.XAVIER)
//                .op
//                .updater(new AdaGrad(learningRate))
                .updater(new Nesterovs(learningRate, 0.5))
                .list()
                .layer(new MessagePassingLayer.Builder().messageActivationFunction(Activation.SIGMOID).updateActivationFunction(Activation.SIGMOID)
                        .nIn(nCol * nRows).nOut(nCol * nRows).build())
                .layer(new MessagePassingLayer.Builder().messageActivationFunction(Activation.SIGMOID).updateActivationFunction(Activation.SIGMOID)
                        .nIn(nCol * nRows).nOut(nCol * nRows).build())
                .layer(new MessagePassingLayer.Builder().messageActivationFunction(Activation.SIGMOID).updateActivationFunction(Activation.SIGMOID)
                        .nIn(nCol * nRows).nOut(nCol * nRows).build())
                .layer(new MessagePassingLayer.Builder().messageActivationFunction(Activation.SIGMOID).updateActivationFunction(Activation.SIGMOID)
                        .nIn(nCol * nRows).nOut(nCol * nRows).build())
                .layer(new ReadoutLayer.Builder().readActivationFunction(Activation.SIGMOID).nIn(nCol * nRows).nOut(2028)
                        .build())
//                .layer(new DenseLayer.Builder().nIn(numHiddenNodes1).nOut(numHiddenNodes1)
//                        .activation(Activation.RELU)
//                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.SOFTMAX)
                        .nIn(2028).nOut(2).build())
                .build();
        //Then add the StatsListener to collect this information from the network, as it trains
        return new MultiLayerNetwork(conf);

    }

    public void startClustering(String dir, String files[]) throws IOException, InterruptedException {
        List<TSNE> l = new ArrayList<>();
        List<INDArray> l1 = generateEmbedding(dir, files);
        INDArray Ge;
        Ge = Nd4j.concat(0, l1.toArray(new INDArray[l1.size()]));
        System.out.println(Ge);
        TSNE tsne = Kmeanstsne(Ge);
        l.add(tsne);
        plot(l);
    }

    private void plot(int[] tsne, INDArray ge) {
        Canvas canvas = ScatterPlot.of(ge.toDoubleMatrix(), tsne, '#').canvas();
//        for (int i = 1; i < tsne.size(); i++) {
//            canvas.add(ScatterPlot.of(tsne.get(i).coordinates, '#', new Color((int) (Math.random() * 0x1000000))));
//        }
        try {
            canvas.window();
        } catch (InvocationTargetException | InterruptedException e) {
            e.printStackTrace();
        }
    }

    public TSNE Kmeanstsne(INDArray Ge) {
//        System.out.println(Ge);
//

//        cluster.
        KMeans cluster = KMeans.fit(Ge.toDoubleMatrix(), 3);
//        cluster.predict()
        TSNE tsne = tsne(Ge.toDoubleMatrix());
//        KMeans cluster = KMeans.fit(tsne.coordinates, 3);
        return tsne;
    }

    private TSNE tsne(double[][] centroids) {
//       centroids= PCA.fit(centroids).setProjection(10).project(centroids);
        return new TSNE(centroids, 2, 50, 500, 1000);
    }

    public TSNE Gmeanstsne(INDArray Ge) {
//        System.out.println(Ge);
//        System.out.println(Ge.columns());
        TSNE tsne = tsne(Ge.toDoubleMatrix());
        GMeans cluster = GMeans.fit(tsne.coordinates, 100, 200, 5);

        return tsne;

    }

    public void plot(List<TSNE> tsne) {
        List<double[][]> l = new ArrayList<>();
        l.add(tsne.get(0).coordinates);

        System.out.println(l);
        int maxIterationCount = 5;
        int clusterCount = 10;
        String distanceFunction = "cosinesimilarity";
//        KMeansClustering kmc = KMeansClustering.setup(clusterCount, maxIterationCount, Distance.COSINE_SIMILARITY,false);
//        ClusterSet cluster = kmc.applyTo(Point.toPoints(Nd4j.create(tsne.get(0).coordinates)));
//        cluster.getClusters().get(0).
//        KMeans cl = PartitionClustering.run(20, () -> KMeans.fit(tsne.get(0).coordinates, 3));
        GMeans cl = GMeans.fit(tsne.get(0).coordinates, 6, 1000, 1.0E-4D);
//        kMeans.
        Canvas canvas = ScatterPlot.of(tsne.get(0).coordinates, cl.y, 'q').canvas();
//        cl.c
//        for (int i = 1; i < tsne.size(); i++) {
//            canvas.add(ScatterPlot.of(tsne.get(i).coordinates, '#', new Color((int) (Math.random() * 0x1000000))));
//        }
        try {
            canvas.window();
        } catch (InvocationTargetException | InterruptedException e) {
            e.printStackTrace();
        }
    }

    private INDArray getGraphEmbedding(String dir, String filename) throws IOException, InterruptedException {
//        DataSet dataset =
//                readData("C:\\Users\\CECSAdmin" +
//                        "\\OneDrive - University of Tennessee at Chattanooga" +
//                        "\\Projects\\ProteinGraph\\exports\\" + filename + ".csv");
        DataSet dataset =
                readData(dir + filename + ".csv");
        List<DataSet> set = dataset.asList();
        System.out.println("input name: " + filename);
        for (int i = 0; i < set.size(); i++) {
            System.out.println("features");
            System.out.println(set.get(i).getFeatures());
            INDArray Hv = Nd4j.zeros(set.get(i).numInputs());

            // get neighbors of i

            ArrayList<Integer> neighbors = new ArrayList<>();
            for (int j = 0; j < set.get(i).getFeatures().columns() - 25; j++) {
                if (set.get(i).getFeatures().getDouble(j) > 0) {
                    if (j != i)
                        neighbors.add(j);
                }
            }
//            for(set.get(i).getLabels())

            // message passing phase
            for (Integer n : neighbors) {
                INDArray message = buildMessagePassingModel(set.get(n.intValue()), set.size());
                Hv = Hv.add(message);
            }
//            for (int j = 0; j < set.size(); j++) { // neighbor
//                if (j != i) {
//                    INDArray message = buildMessagePassingModel(set.get(j), set.size());
//                    Hv = Hv.add(message);
//                }
//            }

            // Update Phase
            Hv.add(set.get(i).getFeatures());
            DataSet set1 = new DataSet(Hv, Hv);
            INDArray update = buildUpdatePhaseModel(set1, set.size());
            set.get(i).setFeatures(update);
//            System.out.println("output");
//            System.out.println(set.get(i).getFeatures());

        }
        //Graph Embedding
        INDArray Ge = Nd4j.zeros(set.get(0).numInputs());
        for (DataSet s : set) {
            Ge = Ge.add(s.getFeatures());
        }
        return getEmbedding(new DataSet(Ge, Ge), 150);
    }

    private INDArray getGraphEmbeddingParallel(String dir, String filename) throws IOException, InterruptedException {
//        DataSet dataset =
//                readData("C:\\Users\\CECSAdmin" +
//                        "\\OneDrive - University of Tennessee at Chattanooga" +
//                        "\\Projects\\ProteinGraph\\exports\\" + filename + ".csv");
        DataSet dataset =
                readData(dir + filename + ".csv");
        List<DataSet> set = dataset.asList();
//        System.out.println("input name: " + filename);
        for (int i = 0; i < iteration; i++) {
            set.parallelStream().forEach((data) -> {
//                    System.out.println("features");
//                    System.out.println(data_30.getFeatures());
                INDArray Hv = Nd4j.zeros(data.numInputs());

                // get neighbors of i
                ArrayList<Integer> neighbors = new ArrayList<>();
                for (int j = 1; j < data.getFeatures().columns() - 25; j++) {
                    if (data.getFeatures().getDouble(j) > 0) {
                        if (j != set.indexOf(data))
                            neighbors.add(j);
                    }
                }
                // message passing phase
                for (Integer n : neighbors) {
                    INDArray message = buildMessagePassingModel(set.get(n.intValue()), set.size());
                    Hv = Hv.add(message);
                }
                // Update Phase
                Hv.add(data.getFeatures());
                DataSet set1 = new DataSet(Hv, Hv);
                INDArray update = buildUpdatePhaseModel(set1, set.size());
                set.get(set.indexOf(data)).setFeatures(update);
            });
        }
        //Graph Embedding
        INDArray Ge = Nd4j.zeros(set.get(0).numInputs());
        Ge = set.parallelStream().reduce(Ge, (ge, data) -> ge.add(data.getFeatures()), INDArray::add);
        return getEmbedding(new DataSet(Ge, Ge), EbeddingNum);
    }


    private INDArray buildMessagePassingModel(DataSet set, int numNodes) { //mxn

        int featureSize = set.numInputs();
//        System.out.println(featureSize);
        MultiLayerConfiguration model = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .l2(0.001)
                .updater(new Nesterovs()) // try other updaters later
                .list()
                .layer(new DenseLayer.Builder().nIn(featureSize).nOut(featureSize)
                        .activation(Activation.RELU)
                        .build())
                .layer(new DenseLayer.Builder().nIn(featureSize).nOut(featureSize)
                        .activation(Activation.RELU)
                        .build())
                .layer(new DenseLayer.Builder().nIn(featureSize).nOut(featureSize)
                        .activation(Activation.RELU)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation(Activation.SOFTMAX)
                        .nIn(featureSize).nOut(featureSize).build())
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(model);
        net.init();
        net.setEpochCount(100);
        net.fit(set);
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
                        .activation(Activation.SIGMOID)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .activation(Activation.IDENTITY)
                        .nIn(featureSize).nOut(featureSize).build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(model);
        net.init();
        net.setEpochCount(100);
        DataSet s = new DataSet();
        net.fit(set);
        return net.output(set.getFeatures());
    }

    private INDArray getEmbedding(DataSet set, int sizes) {
        int featureSize = set.numInputs();
//        System.out.println(featureSize);
        ComputationGraphConfiguration model = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .seed(seed)
                .weightInit(WeightInit.XAVIER)
                .l2(0.001)
                .updater(new AdaDelta()) // try other updaters later
//                .list()

                .graphBuilder()
                .addInputs("input")
                .addLayer("L1", new DenseLayer.Builder().nIn(featureSize).nOut(sizes)
                        .activation(Activation.RELU)
                        .build(), "input")
//                .layer(new DropoutLayer.Builder().dropOut(0.5).nIn(featureSize).nOut(featureSize).build())// Just for test
//                .layer(new DenseLayer.Builder().nIn(30).nOut(35)
//                        .activation(Activation.RELU)
//                        .name("enc2")
//                        .build())
                .addLayer("output", new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX)
                        .nIn(sizes).nOut(featureSize).build(), "L1")
                .setOutputs("output")
                .build();

        ComputationGraph net = new ComputationGraph(model);
        net.init();
        net.fit(set);
        ComputationGraph tl = new TransferLearning.GraphBuilder(net)
                .setFeatureExtractor("L1")
                .removeVertexAndConnections("output")
                .addLayer("output", new OutputLayer.Builder()
                        .nIn(sizes).nOut(sizes)
                        .activation(Activation.IDENTITY).build(), "L1")
                .setOutputs("output")
                .setInputs("input")
                .build();
//        System.out.println(set.getFeatures());
//
        System.out.println(tl.feedForward(set.getFeatures(), false).get("output").columns());
        return tl.feedForward(set.getFeatures(), false).get("output");
    }

    private DataSet readData(String directory) throws IOException, InterruptedException {
        int batchSize = 1500;
        RecordReader rr = new CSVRecordReader();
        NormalizerStandardize normaliser = new NormalizerStandardize();
        rr.initialize(new FileSplit(new File(directory)));

//        rr.

        DataSetIterator iter = new RecordReaderDataSetIterator.Builder(rr, batchSize).build();
        normaliser.fit(iter);
        iter.setPreProcessor(normaliser);
//        new MapFileSequenceRecordReader();
        return iter.next();
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
}
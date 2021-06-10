import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderMultiDataSetIterator;
import org.deeplearning4j.datasets.iterator.IteratorMultiDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.StackVertex;
import org.deeplearning4j.nn.conf.graph.UnstackVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.*;

public class MPNNImp {
    private static ComputationGraph mpnngraph(int numInputs, List<Protein> inputf, long sampledFeatures) {
        int seed = 123;
        double learningRate = 0.5;
        int out = 2; // binary

        String[] inputs = new String[numInputs];
        InputType[] inputsTypes = new InputType[inputf.size()];
        for (int i = 0; i < numInputs; i++) {
            inputs[i] = "input" + String.valueOf(i);
        }

        for (int i = 0; i < inputf.size(); i++) {
            inputsTypes[i] = InputType.inferInputType(inputf.get(i).getFeatureMatrix());
        }
        NeuralNetConfiguration.Builder conf = new NeuralNetConfiguration.Builder();
        ComputationGraphConfiguration.GraphBuilder gbuilder = conf.seed(seed).optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs(learningRate, 0.5))
                .graphBuilder();
//        gbuilder.setInputTypes(inputsTypes);
        gbuilder.addInputs(inputs);
        List<String> outs = new ArrayList<>();
        for (int i = 0; i < inputs.length; i++) {
            gbuilder
                    .addLayer("GCN1" + inputs[i], new MessagePassingLayer.Builder()
                            .messageActivationFunction(Activation.SIGMOID)
                            .adjacencyMatrix(inputf.get(i).getAdjacencyMatrix())
                            .updateActivationFunction(Activation.SIGMOID)
                            .nIn(inputf.get(i).getFeatureMatrix().columns())
                            .nOut(inputf.get(i).getFeatureMatrix().columns())
                            .build(), inputs[i])

                    .addLayer("ROL2" + inputs[i], new ReadoutLayer.Builder()
                            .readActivationFunction(Activation.SIGMOID)
                            .nIn(inputf.get(i).getFeatureMatrix().columns())
                            .nOut(inputf.get(i).getFeatureMatrix().columns())
                            .build(), "GCN1" + inputs[i])
//            outs.add("GCN1" + inputs[i]);
                    .addLayer("DL2" + inputs[i], new DenseLayer.Builder()
                            .nIn(inputf.get(i).getFeatureMatrix().columns())
                            .nOut(sampledFeatures).build(), "ROL2" + inputs[i]);
            outs.add("DL2" + inputs[i]);
//            outs.add("ROL2" + inputs[i]);
        }
        String[] sl = outs.toArray(new String[outs.size()]);

        gbuilder
                .addVertex("stack", new StackVertex(), sl)
                .addLayer("DL3", new DenseLayer.Builder()
                        .nIn(sampledFeatures)
                        .nOut(512).build(), "stack")
                .addVertex("unstack", new UnstackVertex(0, 1), "DL3");
        outs.clear();
//        for (int i = 0; i < inputs.length; i++) {
        gbuilder.addLayer("out", new OutputLayer.Builder()
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.SOFTMAX)
                .nIn(512)
                .nOut(2)
                .build(), "unstack");
//            outs.add("out" + inputs[i]);
//        }


        sl = outs.toArray(new String[outs.size()]);

        ComputationGraphConfiguration conf1 = gbuilder.setOutputs("out").build();
//       System.out.println(conf1.toJson());
        return new ComputationGraph(conf1);
    }

    /*Build the mpnn model with predefine layers for binary classification*/
    private static MultiLayerNetwork mpnn(Protein p, long sampledFeatures) {
        int seed = 123;
        double learningRate = 0.5;
        int out = 2; // binary
        //Configure the layer with custom GCN and Readout layers
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
//                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Nesterovs(learningRate, 0.5))
                .list()
                .layer(new MessagePassingLayer.Builder()
                        .messageActivationFunction(Activation.SIGMOID)
                        .updateActivationFunction(Activation.SIGMOID)
                        .adjacencyMatrix(p.getAdjacencyMatrix())
                        .nIn(p.getFeatureMatrix().columns())
                        .nOut(p.getFeatureMatrix().columns())
                        .build())
                .layer(new ReadoutLayer.Builder()
                        .readActivationFunction(Activation.SIGMOID)
                        .nIn(p.getFeatureMatrix().columns())
                        .nOut(p.getFeatureMatrix().columns())
                        .weightInit(WeightInit.XAVIER)
                        .build())
                .layer(new DenseLayer.Builder()
                        .nIn(p.getFeatureMatrix().columns())
                        .nOut(sampledFeatures)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.RELU)
                        .build()) //Configuring Layers
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.SOFTMAX)
                        .nIn(sampledFeatures)
                        .nOut(2)
                        .build())
                .build();
        return new MultiLayerNetwork(conf);
    }

//    public static void classify(String[] dir, String[][] list) {
//        UIServer uiServer = UIServer.getInstance();
//
//        //Configure where the network information (gradients, score vs. time etc) is to be stored. Here: store in memory.
//        StatsStorage statsStorage = new InMemoryStatsStorage();//Alternative: new FileStatsStorage(File), for saving and loading later
//        //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
//        uiServer.attach(statsStorage);
//        long nRows = 10;// train.numInputs();
//        long nCols = 10;// train.getFeatures().columns();
//        RecordReaderMultiDataSetIterator train = dataset(dir, list);
//        RecordReaderMultiDataSetIterator traint = train;
//        System.out.println("nRows= " + train.getInputs().size() + " nCols= " + nCols);
//        INDArray[] f = new INDArray[train.getInputs().size()];
//        int count = 0;
//        MultiDataSet ds = traint.next();
//        for (INDArray sample : ds.getFeatures()) {
//            f[count] = sample;
////            System.out.println(sample.columns());
//            count++;
//        }
//        INDArray[] sample = ds.getLabels();
//        INDArray[] labels = new INDArray[train.getInputs().size()];
//        for (int i = 0; i < sample.length; i++) {
//            labels[i] = sample[i].getRow(0);
//        }
//        System.out.println();
//
////        ComputationGraph model = mpnngraph(train.getInputs().size(), f, 2028);
////
////
////        model.setListeners(new StatsListener(statsStorage));
////        model.init();
////        INDArray[] features = ds.getFeatures();
////        Map<String, INDArray> r = model.feedForward(features, true);
////        for (Map.Entry<String, INDArray> e : r.entrySet()
////        ) {
////            if (e.getKey().equalsIgnoreCase("merge"))
////                System.out.println(e.getValue());
////        }
//    }

    public static void classify() throws IOException, ClassNotFoundException {
        ProteinsDataset dataset = Utils.readFile();
        List<Protein> p = dataset.getDataset();
        int ins = p.size() - 1;
        INDArray[] inputs = new INDArray[ins];
        INDArray[] adj = new INDArray[ins];
//        INDArray[] labels = new INDArray[p.size()];
        INDArray labels = Nd4j.zeros(ins, 2);
        NormalizerMinMaxScaler normalizer = new NormalizerMinMaxScaler();


        for (int i = 0; i < ins; i++) {
            INDArray in = p.get(i).getFeatureMatrix();
            INDArray ad = p.get(i).getAdjacencyMatrix();
            INDArray out = p.get(i).getLabel();
            normalizer.setFeatureStats(Nd4j.create(1).add(0), Nd4j.create(1).add(1));
            normalizer.transform(in);
            normalizer.transform(out);
            inputs[i] = in;
            adj[i] = ad;
//            for (int j = 0; j < in.rows(); j++) {
//                labels.add(out);
//            }
            labels.putScalar(i, 0, out.getDouble(0));
            labels.putScalar(i, 1, out.getDouble(1));
        }
//        System.out.println(labels);
//        ComputationGraph model = mpnngraph(ins, p, 100);
        MultiLayerNetwork model = mpnn(p.get(0), 2012);
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        model.setListeners(new StatsListener(statsStorage));
        model.init();
//        System.out.println(p.get(0).getFeatureMatrix());
//        MultiDataSet dataSet=new MultiDataSet(inputs, labels);
//        MultiDataSetIterator iter = new IteratorMultiDataSetIterator
//                (Collections.singletonList((org.nd4j.linalg.dataset.api.MultiDataSet) dataSet).iterator(), 1500);

        for (int i = 0; i < 1; i++) {
//            model.fit(new MultiDataSet(inputs, new INDArray[] { labels }));
            model.feedForward(inputs[0],true);
            model.feedForward(inputs[0],true);

        }
//        ComputationGraph tl = new TransferLearning.GraphBuilder(net)
//                .setFeatureExtractor("L1")
//                .removeVertexAndConnections("output")
//                .addLayer("output", new OutputLayer.Builder()
//                        .nIn(sizes).nOut(sizes)
//                        .activation(Activation.IDENTITY).build(), "L1")
//                .setOutputs("output")
//                .setInputs("input")
//                .build();

//        model.feedForward(inputs,true);
//        model.fit(inputs, labels.toArray(new INDArray[labels.size()]));
//        Map<String, INDArray> r = model.feedForward(inputs, true);
//        List<INDArray> r = model.feedForward(p.get(0).getFeatureMatrix());
//        System.out.println("Hi: "+ r.get(4));
//        for (Map.Entry<String, INDArray> e : r.entrySet()
//        ) {
//            if (e.getKey().equalsIgnoreCase("out"))
//                System.out.println(e.getValue());
//        }

//        for (int i = 0; i < ins; i++) {
//            INDArray in = p.get(i).getFeatureMatrix();
//            INDArray ad = p.get(i).getAdjacencyMatrix();
//            INDArray out = p.get(i).getLabel();
//            normalizer.setFeatureStats(Nd4j.create(1).add(0), Nd4j.create(1).add(1));
//            normalizer.transform(in);
//            normalizer.transform(out);
//            labels.putScalar(i,0,out.getDouble(0));
//            labels.putScalar(i,1,out.getDouble(1));
//            model.fit();
//        }
//        Evaluation eval = new Evaluation(2);
//        INDArray[] output = model.output(p.get(ins).getFeatureMatrix());
//        eval.eval(p.get(ins).getLabel(), output[0]);
////        log.info(eval.stats());

//        System.out.println(eval.stats());
    }

    private static RecordReaderMultiDataSetIterator dataset(String[] dir, String[][] list) {
        Map<String, RecordReader> rrs = getRR(dir, list);
//
        RecordReaderMultiDataSetIterator.Builder builder = new RecordReaderMultiDataSetIterator.Builder(1);
//
        for (Map.Entry<String, RecordReader> e : rrs.entrySet()) {

            builder.addReader(e.getKey(), e.getValue());
            builder.addInput(e.getKey());
            builder.addOutputOneHot(e.getKey(), 0, 2);


        }
        RecordReaderMultiDataSetIterator b = builder.build();
        return b;

    }

    private static Map<String, RecordReader> getRR(String[] dir, String[][] list) {
        String dir1 = dir[0];
        List<String> list1 = Arrays.asList(list[0]);
        String dir2 = dir[1];
        List<String> list2 = Arrays.asList(list[1]);

        HashMap<String, RecordReader> m = new HashMap<>();

        list1.parallelStream().forEach((filename) -> {
            String path = dir1 + filename + ".csv";
            RecordReader rr = new CSVRecordReader();
            try {
                rr.initialize(new FileSplit(new File(path)));
                m.put(filename, rr);
            } catch (IOException | InterruptedException e) {
                e.printStackTrace();
            }
        });
//        for(String filename:list[0]){
//            String path = dir1 + filename + ".csv";
//            RecordReader rr = new CSVRecordReader();
//            try {
//                rr.initialize(new FileSplit(new File(path)));
//                m.put(filename, rr);
//
//            } catch (IOException | InterruptedException e) {
//                e.printStackTrace();
//            }
//        }
//        for(String filename:list[1]){
//            String path = dir2 + filename + ".csv";
//            RecordReader rr = new CSVRecordReader();
//            try {
//                rr.initialize(new FileSplit(new File(path)));
//                m.put(filename, rr);
//            } catch (IOException | InterruptedException e) {
//                e.printStackTrace();
//            }
//        }
        list2.parallelStream().forEach((filename) -> {
            String path = dir2 + filename + ".csv";
            RecordReader rr = new CSVRecordReader();
            try {
                rr.initialize(new FileSplit(new File(path)));
                m.put(filename, rr);
            } catch (IOException | InterruptedException e) {
                e.printStackTrace();
            }
        });
        return m;
    }
}

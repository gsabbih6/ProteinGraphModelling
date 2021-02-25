import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderMultiDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.graph.MergeVertex;
import org.deeplearning4j.nn.conf.graph.StackVertex;
import org.deeplearning4j.nn.conf.graph.UnstackVertex;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.dataset.api.MultiDataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.*;

public class MPNNImp {
    private static ComputationGraph mpnngraph(int numInputs, INDArray[] inputf, long sampledFeatures) {
        int seed = 123;
        double learningRate = 0.5;
        int out = 2; // binary

        String[] inputs = new String[numInputs];
        InputType[] inputsTypes = new InputType[inputf.length];
        for (int i = 0; i < numInputs; i++) {
            inputs[i] = "input" + String.valueOf(i);
        }

        for (int i = 0; i < inputf.length; i++) {
            inputsTypes[i] = InputType.convolutional(inputf[i].rows(), inputf[i].columns(), 1);
        }
        NeuralNetConfiguration.Builder conf = new NeuralNetConfiguration.Builder();
        ComputationGraphConfiguration.GraphBuilder gbuilder = conf.seed(seed).optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs(learningRate, 0.5))
                .graphBuilder();
        gbuilder.setInputTypes(inputsTypes);
        gbuilder.addInputs(inputs);
        List<String> outs = new ArrayList<>();
        for (int i = 0; i < inputs.length; i++) {
            gbuilder
                    .addLayer("GCN1" + inputs[i], new MessagePassingLayer.Builder().messageActivationFunction(Activation.SIGMOID).updateActivationFunction(Activation.SIGMOID)
                            .nIn(inputf[i].columns())
                            .nOut(inputf[i].columns())
                            .build(), inputs[i])
                    .addLayer("ROL2" + inputs[i], new ReadoutLayer.Builder().readActivationFunction(Activation.SIGMOID)
                            .nIn(inputf[i].columns())
                            .nOut(sampledFeatures)
                            .build(), "GCN1" + inputs[i]);
//            outs.add("GCN1" + inputs[i]);
            outs.add("ROL2" + inputs[i]);
        }
        String[] sl = outs.toArray(new String[outs.size()]);
//        System.out.println(sl[0]);
        gbuilder
                .addVertex("merge", new StackVertex(), sl)
//                .addVertex("merge1", new UnstackVertex(0,15), "merge")
                .addLayer("DL1", new DenseLayer.Builder().nIn(sampledFeatures).nOut(sampledFeatures).build(), "merge")
                .addLayer("DL2", new DenseLayer.Builder().nIn(sampledFeatures).nOut(512).build(), "DL1");
        outs.clear();
        for (int i = 0; i < inputs.length; i++) {
            gbuilder.addLayer("out" + inputs[i], new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                    .weightInit(WeightInit.XAVIER)
                    .activation(Activation.SOFTMAX)
                    .nIn(512)
                    .nOut(2)
                    .build(), "DL2");
            outs.add("out" + inputs[i]);
        }
        sl = outs.toArray(new String[outs.size()]);
        ComputationGraphConfiguration conf1 = gbuilder.setOutputs(sl).build();


        //Configure the layer with custom GCN and Readout layers
//        ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
//                .seed(seed)
//                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
//                .weightInit(WeightInit.XAVIER)
//                .updater(new Nesterovs(learningRate, 0.5))
//                .graphBuilder()
//                .setInputTypes(inputsTypes)
//                .addInputs(inputs)
//                .addLayer("GCN1", new MessagePassingLayer.Builder().messageActivationFunction(Activation.SIGMOID).updateActivationFunction(Activation.SIGMOID)
//                        .build(), inputs)
//                .addLayer("GCN2", new MessagePassingLayer.Builder().messageActivationFunction(Activation.SIGMOID).updateActivationFunction(Activation.SIGMOID)
//                        .build(), "GCN1")
//                .addLayer("GCN3", new MessagePassingLayer.Builder().messageActivationFunction(Activation.SIGMOID).updateActivationFunction(Activation.SIGMOID)
//                        .build(), "GCN2")
//                .addLayer("GCN4", new MessagePassingLayer.Builder().messageActivationFunction(Activation.SIGMOID).updateActivationFunction(Activation.SIGMOID)
//                        .build(), "GCN3")
////                .addLayer("GCN4", new MessagePassingLayer.Builder().messageActivationFunction(Activation.SIGMOID).updateActivationFunction(Activation.SIGMOID)
////                        .nIn(sampledFeatures)
////                        .nOut(sampledFeatures)
////                        .build(), inputs)
//                .addLayer("ROL", new ReadoutLayer.Builder().readActivationFunction(Activation.SIGMOID)
//                        .nOut(sampledFeatures)
//                        .build(), "GCN4")
//
//                .addLayer("DL1", new DenseLayer.Builder().nIn(sampledFeatures).nOut(sampledFeatures).build(), "ROL")
//                .addLayer("DL2", new DenseLayer.Builder().nIn(sampledFeatures).nOut(sampledFeatures - 1000).build(), "DL1")
//                .addLayer("out", new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
//                        .weightInit(WeightInit.XAVIER)
//                        .activation(Activation.SOFTMAX)
//                        .nIn(sampledFeatures)
//                        .nIn(sampledFeatures)
//                        .build(), "DL2")
//                .setOutputs("out")
//                .build();
        return new ComputationGraph(conf1);
    }

    /*Build the mpnn model with predefine layers for binary classification*/
    private static MultiLayerNetwork mpnn(long nRows, long nCol, long sampledFeatures) {
        int seed = 123;
        double learningRate = 0.5;
        int out = 2; // binary
        //Configure the layer with custom GCN and Readout layers
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs(learningRate, 0.5))
                .list()
                .layer(new MessagePassingLayer.Builder().messageActivationFunction(Activation.SIGMOID).updateActivationFunction(Activation.SIGMOID)
                        .nIn(nCol * nRows).nOut(nCol * nRows)
                        .build())
                .layer(new MessagePassingLayer.Builder().messageActivationFunction(Activation.SIGMOID).updateActivationFunction(Activation.SIGMOID)
                        .nIn(nCol * nRows).nOut(nCol * nRows)
                        .build())
                .layer(new MessagePassingLayer.Builder().messageActivationFunction(Activation.SIGMOID).updateActivationFunction(Activation.SIGMOID)
                        .nIn(nCol * nRows).nOut(nCol * nRows)
                        .build())
                .layer(new MessagePassingLayer.Builder().messageActivationFunction(Activation.SIGMOID).updateActivationFunction(Activation.SIGMOID)
                        .nIn(nCol * nRows).nOut(nCol * nRows)
                        .build())
//                .layer(new ReadoutLayer.Builder().readActivationFunction(Activation.SIGMOID)
//                        .nIn(nCol * nRows)
//                        .nOut(sampledFeatures)
//                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.SOFTMAX)
//                        .nIn(sampledFeatures)
                        .nIn(nCol * nRows)
                        .nOut(out).build())
                .build();
        return new MultiLayerNetwork(conf);
    }

    public static void classify(String[] dir, String[][] list) {
        UIServer uiServer = UIServer.getInstance();

        //Configure where the network information (gradients, score vs. time etc) is to be stored. Here: store in memory.
        StatsStorage statsStorage = new InMemoryStatsStorage();//Alternative: new FileStatsStorage(File), for saving and loading later
        //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
        uiServer.attach(statsStorage);
        long nRows = 10;// train.numInputs();
        long nCols = 10;// train.getFeatures().columns();
        RecordReaderMultiDataSetIterator train = dataset(dir, list);
        RecordReaderMultiDataSetIterator traint = train;
        System.out.println("nRows= " + train.getInputs().size() + " nCols= " + nCols);
        INDArray[] f = new INDArray[train.getInputs().size()];
        int count = 0;
        MultiDataSet ds = traint.next();
        for (INDArray sample : ds.getFeatures()) {
            f[count] = sample;
//            System.out.println(sample.columns());
            count++;
        }
        INDArray[] sample = ds.getLabels();
        INDArray[] labels = new INDArray[train.getInputs().size()];
        for (int i = 0; i < sample.length; i++) {
            labels[i] = sample[i].getRow(0);
        }
        System.out.println();

        ComputationGraph model = mpnngraph(train.getInputs().size(), f, 2028);


        model.setListeners(new StatsListener(statsStorage));
        model.init();


//        System.out.println(tra);
//        ds.setLabels(labels);
        INDArray[] features = ds.getFeatures();
//        INDArray[] labels = trainl.next().getFeatures();

//        for (int i = 0; i < 5000; i++) {
////            model.fit(features,labels);
////            model.fit(ds);
////            model.feedForward(features,true);
//        }
        Map<String, INDArray> r = model.feedForward(features, true);
        for (Map.Entry<String, INDArray> e:r.entrySet()
             ) {
            if(e.getKey().equalsIgnoreCase("merge"))
            System.out.println(e.getValue());
        }


//        System.out.println("\n**************** Example finished ********************");
//        model.save(new File("C:\\Users\\CECSAdmin\\OneDrive - University of Tennessee at Chattanooga" +
//                "\\Projects\\ProteinGraph\\exports\\model_150"));
//
//evaluate the model_30 on the test set
//    Evaluation eval = new Evaluation(2);
//    INDArray output = model.output(test.getFeatures());
//    eval.eval(test.getLabels(), output);
//        log.info(eval.stats());

//    System.out.println(eval.stats());
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

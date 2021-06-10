import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.nd4j.autodiff.listeners.At;
import org.nd4j.autodiff.listeners.Listener;
import org.nd4j.autodiff.listeners.ListenerResponse;
import org.nd4j.autodiff.listeners.Loss;
import org.nd4j.autodiff.listeners.impl.HistoryListener;
import org.nd4j.autodiff.listeners.impl.ScoreListener;
import org.nd4j.autodiff.listeners.impl.UIListener;
import org.nd4j.autodiff.listeners.records.EvaluationRecord;
import org.nd4j.autodiff.listeners.records.History;
import org.nd4j.autodiff.listeners.records.LossCurve;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.evaluation.IEvaluation;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.evaluation.classification.ROCBinary;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.MultiDataSet;
import org.nd4j.linalg.dataset.api.iterator.MultiDataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.AdaGrad;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.weightinit.impl.ReluInitScheme;
import org.nd4j.weightinit.impl.XavierInitScheme;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class SameDiffGCN {
    SameDiff sd = SameDiff.create();

    ProteinsDataset p = null;

    //set placeholders for
    public void creatModel(int featureSize, int outputSize) throws IOException {
        try {
            p = Utils.readFile();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }

        featureSize = p.getDataset().get(0).getFeatureMatrix().columns();
        outputSize = p.getDataset().get(0).getLabel().columns();
        System.out.println("featureSize:" + featureSize + "outputSize: " + outputSize);
        SDVariable feature = sd.placeHolder("feature", DataType.FLOAT, -1, featureSize);
        SDVariable adjacency = sd.placeHolder("adjacency", DataType.FLOAT, -1, -1);
        SDVariable clique = sd.placeHolder("clique", DataType.FLOAT, 1, outputSize);
        SDVariable labels = sd.placeHolder("label", DataType.FLOAT, 1, outputSize);

        //mp layer pass
        SDVariable w0 = sd.var("w0", new ReluInitScheme('c', featureSize), DataType.FLOAT, featureSize, featureSize);
        SDVariable b0 = sd.zero("b0");
        SDVariable w1 = sd.var("w1", new ReluInitScheme('c', featureSize), DataType.FLOAT, featureSize, featureSize);
        SDVariable b1 = sd.zero("b1");
        SDVariable w2 = sd.var("w2", new ReluInitScheme('c', featureSize), DataType.FLOAT, featureSize, featureSize);
        SDVariable b2 = sd.zero("b2");
        SDVariable w3 = sd.var("w3", new ReluInitScheme('c', featureSize), DataType.FLOAT, featureSize, 500);
        SDVariable b3 = sd.zero("b3");
        SDVariable w4 = sd.var("w4", new ReluInitScheme('c', 500), DataType.FLOAT, 500, 1024);
        SDVariable b4 = sd.zero("b4");
        SDVariable w5 = sd.var("w5", new ReluInitScheme('c', 1024), DataType.FLOAT, 1024, 512);
        SDVariable b5 = sd.zero("b5");
        SDVariable w6 = sd.var("w6", new ReluInitScheme('c', 512), DataType.FLOAT, 512, outputSize);
        SDVariable b6 = sd.zero("b6");

//        SDVariable di = sd.math.diag(adjacency.sum(1).pow(-0.5));
//        adjacency = di.mmul(adjacency).mmul(di);

        //3 GCN Layers
        SDVariable z0 = adjacency.mmul("premul0", feature).mmul("msgmul0", w0);//.add(b0);
        SDVariable a0 = sd.nn().relu("softmax0", z0, 0);
        SDVariable z1 = adjacency.mmul("premul1", a0).mmul("msgmul1", w1);//.add(b1);
        SDVariable a1 = sd.nn().relu("softmax1", z1, 0).add(a0);
        SDVariable z2 = adjacency.mmul("premul2", a1).mmul("msgmul2", w2);//.add(b2);
        SDVariable a2 = sd.nn().relu("softmax2", z2, 0).add(a1);//.add(a0);

        //update layer pass
        SDVariable z3 = a2.mmul("premul3", w3).add(b3);
//        z3 = sd.sum("merge", z3, true, 0);
//sd.cnn.max
        SDVariable a3 = sd.sum("merge", sd.nn().relu("softmax3", z3, 0), true, 0);


        //Dense Layer
        SDVariable z4 = a3.mmul("dense0", w4).add(b4);
        SDVariable a4 = sd.nn().relu("softmax4", z4, 0);
        SDVariable z5 = a4.mmul("dense1", w5).add(b5);
        SDVariable a5 = sd.nn().relu("softmax5", z5, 0);
        SDVariable z6 = a5.mmul("dense2", w6).add(b6);
        SDVariable a6 = sd.nn().sigmoid("softmax6", z6);

        //Define loss function:
//        SDVariable diff = sd.math.squaredDifference("siglos", a6, labels);
        SDVariable diff = sd.loss.sigmoidCrossEntropy("sqd", labels, a6, null);
//        SDVariable lossMse = diff.mean("mse");
//
        sd.setLossVariables(diff);
        train();
    }


    public void creatModel1(int featureSize, int outputSize) throws IOException {
        try {
            p = Utils.readFile();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }

        featureSize = p.getDataset().get(0).getFeatureMatrix().columns();
        outputSize = p.getDataset().get(0).getLabel().columns();
        System.out.println("featureSize:" + featureSize + "outputSize: " + outputSize);
        SDVariable feature = sd.placeHolder("feature", DataType.FLOAT, -1, featureSize);
        SDVariable adjacency = sd.placeHolder("adjacency", DataType.FLOAT, -1, -1);
        SDVariable labels = sd.placeHolder("label", DataType.FLOAT, 1, outputSize);

        //mp layer pass
        SDVariable w0 = sd.var("w0", new ReluInitScheme('c', featureSize), DataType.FLOAT, featureSize, featureSize);
        SDVariable b0 = sd.zero("b0");
        SDVariable w1 = sd.var("w1", new ReluInitScheme('c', featureSize), DataType.FLOAT, featureSize, featureSize);
        SDVariable b1 = sd.zero("b1");
        SDVariable w2 = sd.var("w2", new ReluInitScheme('c', featureSize), DataType.FLOAT, featureSize, featureSize);
        SDVariable b2 = sd.zero("b2");
        SDVariable w3 = sd.var("w3", new ReluInitScheme('c', featureSize), DataType.FLOAT, featureSize, 1024);
        SDVariable b3 = sd.zero("b3");
        SDVariable w6 = sd.var("w6", new ReluInitScheme('c', 1024), DataType.FLOAT, 1024, outputSize);
        SDVariable b6 = sd.zero("b6");

        //2 GCN Layers with skip connections
        SDVariable z0 = adjacency.mmul("premul0", feature).mmul("msgmul0", w0);//.add(b0);
        SDVariable a0 = sd.nn().relu("softmax0", z0, 0);
        SDVariable z1 = adjacency.mmul("premul1", a0).mmul("msgmul1", w1);//.add(b1);
        SDVariable a1 = sd.nn().relu("softmax1", z1, 0);//.add(a0);

        //update layer pass
        SDVariable z3 = a1.mmul("premul3", w3).add(b3);
        SDVariable a3 = sd.sum("merge", sd.nn().relu("softmax3", z3, 0), true, 0);


        //one Dense Layer
        SDVariable z6 = a3.mmul("dense2", w6).add(b6);
        SDVariable a6 = sd.nn().sigmoid("softmax6", z6);

        //Define loss function:
//        SDVariable diff = sd.math.squaredDifference("siglos", a6, labels);
        SDVariable diff = sd.loss.sigmoidCrossEntropy("sqd", labels, a6, null);
//        SDVariable lossMse = diff.mean("mse");
//
        sd.setLossVariables(diff);
        train();
    }


    public void creatModel3(int featureSize, int outputSize) throws IOException {
        try {
            p = Utils.readFile();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }

        featureSize = p.getDataset().get(0).getFeatureMatrix().columns();
        outputSize = p.getDataset().get(0).getLabel().columns();
        System.out.println("featureSize:" + featureSize + "outputSize: " + outputSize);
        SDVariable feature = sd.placeHolder("feature", DataType.FLOAT, -1, featureSize);
        SDVariable adjacency = sd.placeHolder("adjacency", DataType.FLOAT, -1, -1);
        SDVariable labels = sd.placeHolder("label", DataType.FLOAT, 1, outputSize);

        //mp layer pass
        SDVariable w0 = sd.var("w0", new XavierInitScheme('c', featureSize, featureSize), DataType.FLOAT, featureSize, featureSize);
        SDVariable b0 = sd.zero("b0");
        SDVariable w1 = sd.var("w1", new XavierInitScheme('c', featureSize, featureSize), DataType.FLOAT, featureSize, featureSize);
        SDVariable b1 = sd.zero("b1");
        SDVariable w2 = sd.var("w2", new XavierInitScheme('c', featureSize, featureSize), DataType.FLOAT, featureSize, featureSize);
        SDVariable b2 = sd.zero("b2");
        SDVariable w3 = sd.var("w3", new XavierInitScheme('c', featureSize, featureSize), DataType.FLOAT, featureSize, featureSize);
        SDVariable b3 = sd.zero("b3");
        SDVariable w4 = sd.var("w4", new XavierInitScheme('c', featureSize, featureSize), DataType.FLOAT, featureSize, featureSize);
        SDVariable b4 = sd.zero("b4");
        SDVariable w5 = sd.var("w5", new XavierInitScheme('c', featureSize, featureSize), DataType.FLOAT, featureSize, featureSize);
        SDVariable b5 = sd.zero("b5");
        SDVariable w6 = sd.var("w6", new XavierInitScheme('c', featureSize, featureSize), DataType.FLOAT, featureSize, outputSize);
        SDVariable b6 = sd.zero("b6");

        //5 GCN Layers with skip connections
        SDVariable z0 = adjacency.mmul("premul0", feature).mmul("msgmul0", w0);//.add(b0);
        SDVariable a0 = sd.nn().relu("softmax0", z0, 0);
        SDVariable z1 = adjacency.mmul("premul1", a0).mmul("msgmul1", w1);//.add(b1);
        SDVariable a1 = sd.nn().relu("softmax1", z1, 0).add(a0);
        SDVariable z2 = adjacency.mmul("premul2", a1).mmul("msgmul2", w2);//.add(b1);
        SDVariable a2 = sd.nn().relu("softmax2", z2, 0).add(a1);
        SDVariable z3 = adjacency.mmul("premul3", a2).mmul("msgmul3", w3);//.add(b1);
        SDVariable a3 = sd.nn().relu("softmax3", z3, 0).add(a2);
        SDVariable z4 = adjacency.mmul("premul4", a3).mmul("msgmul4", w4);//.add(b1);
        SDVariable a4 = sd.nn().relu("softmax4", z4, 0).add(a3);

        //update layer pass
        SDVariable z5 = a4.mmul("premul5", w5).add(b5);
        z5 = sd.sum("merge", z5, true, 0);

        SDVariable a5 = sd.nn().relu("softmax5", z5, 0);


        //one Dense Layer
        SDVariable z6 = a5.mmul("dense1", w6).add(b6);
        SDVariable a6 = sd.nn().softmax("softmax6", z6);

        //Define loss function:
        SDVariable diff = sd.math.squaredDifference("siglos", a6, labels);
//        SDVariable diff = sd.loss.hingeLoss("sqd", labels,a6, null);
        SDVariable lossMse = diff.mean("mse");
//
        sd.setLossVariables(lossMse);
        train();
    }

    public void creatModel4(int featureSize, int outputSize) throws IOException {
        try {
            p = Utils.readFile();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }

        featureSize = p.getDataset().get(0).getFeatureMatrix().columns();
        outputSize = p.getDataset().get(0).getLabel().columns();
        System.out.println("featureSize:" + featureSize + "outputSize: " + outputSize);
        SDVariable feature = sd.placeHolder("feature", DataType.FLOAT, -1, featureSize);
        SDVariable adjacency = sd.placeHolder("adjacency", DataType.FLOAT, -1, -1);
        SDVariable labels = sd.placeHolder("label", DataType.FLOAT, 1, outputSize);

        //mp layer pass
        SDVariable w0 = sd.var("w0", new ReluInitScheme('c', feature.getShape()[0]), DataType.FLOAT, featureSize, featureSize);
        SDVariable b0 = sd.zero("b0");
        SDVariable w1 = sd.var("w1", new ReluInitScheme('c', feature.getShape()[0]), DataType.FLOAT, featureSize, featureSize);
        SDVariable b1 = sd.zero("b1");
        SDVariable w2 = sd.var("w2", new ReluInitScheme('c', feature.getShape()[0]), DataType.FLOAT, featureSize, featureSize);
        SDVariable b2 = sd.zero("b2");
        SDVariable w3 = sd.var("w3", new ReluInitScheme('c', feature.getShape()[0]), DataType.FLOAT, featureSize, 2015);
        SDVariable b3 = sd.zero("b3");
        SDVariable w4 = sd.var("w4", new ReluInitScheme('c', featureSize), DataType.FLOAT, featureSize, featureSize);
        SDVariable b4 = sd.zero("b4");
        SDVariable w5 = sd.var("w5", new ReluInitScheme('c', 1), DataType.FLOAT, 2015, 512);
        SDVariable b5 = sd.zero("b5");
        SDVariable w6 = sd.var("w6", new ReluInitScheme('c', 1), DataType.FLOAT, 512, outputSize);
        SDVariable b6 = sd.zero("b6");

        SDVariable di = sd.math.diag(adjacency.sum(1).pow(-0.5));
        adjacency = di.mmul(adjacency).mmul(di);

        //2 GCN Layers with skip connections
        SDVariable z0 = adjacency.mmul("premul0", feature).mmul("msgmul0", w0);//.add(b0);
        SDVariable a0 = sd.nn().relu("softmax0", z0, 0);
        SDVariable z1 = adjacency.mmul("premul1", a0).mmul("msgmul1", w1);//.add(b1);
        SDVariable a1 = sd.nn().relu("softmax1", z1, 0).add(a0);

        //update layer pass
        SDVariable z3 = sd.sum("merge", a1.mmul(w3), true, 0);
        SDVariable a3 = sd.nn.relu(z3, 0);


        //one Dense Layer
        SDVariable z5 = a3.mmul("dense1", w5).add(b5);
        SDVariable a5 = sd.nn().relu("softmax5", z5, 0);
        SDVariable z6 = a5.mmul("dense2", w6).add(b6);
        SDVariable a6 = sd.nn().sigmoid("softmax6", z6);

        //Define loss function:
//        SDVariable diff = sd.math.squaredDifference("siglos", a6, labels);
        SDVariable diff = sd.loss.sigmoidCrossEntropy("sqd", labels, a6, null);
//        SDVariable lossMse = diff.mean("mse");
//
        sd.setLossVariables(diff);
        train();
    }

    public void creatModel5(int featureSize, int outputSize) throws IOException {
        try {
            p = Utils.readFile();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }

        featureSize = p.getDataset().get(0).getFeatureMatrix().columns();
        outputSize = p.getDataset().get(0).getLabel().columns();
        System.out.println("featureSize:" + featureSize + "outputSize: " + outputSize);
        SDVariable feature = sd.placeHolder("feature", DataType.FLOAT, -1, featureSize);
        SDVariable adjacency = sd.placeHolder("adjacency", DataType.FLOAT, -1, -1);
        SDVariable labels = sd.placeHolder("label", DataType.FLOAT, 1, outputSize);

        //mp layer pass
        SDVariable w0 = sd.var("w0", new ReluInitScheme('c', featureSize), DataType.FLOAT, featureSize, featureSize);
        SDVariable b0 = sd.zero("b0");
        SDVariable w1 = sd.var("w1", new ReluInitScheme('c', featureSize), DataType.FLOAT, featureSize, featureSize);
        SDVariable b1 = sd.zero("b1");
        SDVariable w2 = sd.var("w2", new ReluInitScheme('c', featureSize), DataType.FLOAT, featureSize, featureSize);
        SDVariable b2 = sd.zero("b2");
        SDVariable w3 = sd.var("w3", new ReluInitScheme('c', featureSize), DataType.FLOAT, featureSize, featureSize);
        SDVariable b3 = sd.zero("b3");
        SDVariable w4 = sd.var("w4", new ReluInitScheme('c', featureSize), DataType.FLOAT, featureSize, featureSize);
        SDVariable b4 = sd.zero("b4");
        SDVariable w5 = sd.var("w5", new ReluInitScheme('c', featureSize), DataType.FLOAT, featureSize, featureSize);
        SDVariable b5 = sd.zero("b5");
        SDVariable w6 = sd.var("w6", new ReluInitScheme('c', featureSize), DataType.FLOAT, featureSize, outputSize);
        SDVariable b6 = sd.zero("b6");

        SDVariable di = sd.math.diag(adjacency.sum(1).pow(-0.5));
        adjacency = di.mmul(adjacency).mmul(di);

        //2 GCN Layers with skip connections
        SDVariable z0 = adjacency.mmul("premul0", feature).mmul("msgmul0", w0);//.add(b0);
        SDVariable a0 = sd.nn().relu("softmax0", z0, 0);
        SDVariable z1 = adjacency.mmul("premul1", a0).mmul("msgmul1", w1);//.add(b1);
        SDVariable a1 = sd.nn().relu("softmax1", z1, 0).add(a0);

        //update layer pass
        SDVariable a3 = sd.nn.relu(sd.sum("merge", a1, true, 0), 0);


        //one Dense Layer
        SDVariable z6 = a3.mmul("dense2", w6).add(b6);
        SDVariable a6 = sd.nn().softmax("softmax6", z6);

        //Define loss function:
//        SDVariable diff = sd.math.squaredDifference("siglos", a6, labels);
        SDVariable diff = sd.loss.softmaxCrossEntropy("sqd", labels, a6, null);
//        SDVariable lossMse = diff.mean("mse");
//
        sd.setLossVariables(diff);
        train();
    }

    public SameDiff createModel(int nGCNLayers, int nDenseLayer, int GCNActivation, int denseLayerActivation, int outputActivation, int lossFunction) {


        return null;
    }

    private void train() throws IOException {
        double learningRate = 1e-3;
        TrainingConfig config = new TrainingConfig.Builder()
                .l2(1e-4)
                //L2 regularization
                .updater(new Adam(learningRate))        //Adam optimizer with specified learning rate
                .dataSetFeatureMapping("feature", "adjacency")         //DataSet features array should be associated with variable "input"
                .dataSetLabelMapping("label")
//                .validationEvaluation("softmax6", 0, new IEvaluation[]{new ROCBinary()})//DataSet label array should be associated with variable "label"
//                .addEvaluations(false, "softmax6", 0, new Evaluation())
                .build();

        sd.setTrainingConfig(config);

//        System.out.println(sd.summary());
        List<org.nd4j.linalg.dataset.api.MultiDataSet> trainList = new ArrayList<>();
        List<org.nd4j.linalg.dataset.api.MultiDataSet> testList = new ArrayList<>();
        List<Protein> data = p.getDataset();
        Collections.shuffle(data);// randomise data
        Collections.shuffle(data);
        Collections.shuffle(data);
        double fraction = 0.95;
        int num = (int) (fraction * data.size());
        NormalizerMinMaxScaler normalizer = new NormalizerMinMaxScaler();
        for (int i = 0; i < num; i++) {
            INDArray in = p.getDataset().get(i).getFeatureMatrix();
            normalizer.setFeatureStats(Nd4j.create(1).add(0), Nd4j.create(1).add(1));
            normalizer.transform(in);
            org.nd4j.linalg.dataset.api.MultiDataSet ds = new MultiDataSet();
            INDArray di = Nd4j.diag(Transforms.pow(p.getDataset().get(i).getAdjacencyMatrix().sum(1), -0.5));
            INDArray adjacency = di.mmul(p.getDataset().get(i).getAdjacencyMatrix()).mmul(di);
            ds.setFeatures(new INDArray[]{in, adjacency});
            ds.setLabels(new INDArray[]{p.getDataset().get(i).getLabel()});
            trainList.add(ds);
        }


        for (int i = num; i < data.size(); i++) {
            INDArray in = p.getDataset().get(i).getFeatureMatrix();
            normalizer.setFeatureStats(Nd4j.create(1).add(0), Nd4j.create(1).add(1));
            normalizer.transform(in);
            org.nd4j.linalg.dataset.api.MultiDataSet ds = new MultiDataSet();
            INDArray di = Nd4j.diag(Transforms.pow(p.getDataset().get(i).getAdjacencyMatrix().sum(1), -0.5));
            INDArray adjacency = di.mmul(p.getDataset().get(i).getAdjacencyMatrix()).mmul(di);
            ds.setFeatures(new INDArray[]{in, adjacency});
            ds.setLabels(new INDArray[]{p.getDataset().get(i).getLabel()});
            testList.add(ds);
        }
//        MultiNormalizerStandardize normalizer = new MultiNormalizerStandardize();
//        MultiNormalizerStandardize normalizer1 = new MultiNormalizerStandardize();
        MultiDataSetIterator traindata = new MyIteratorD(trainList, 2);

        MultiDataSetIterator testdata = new MyIteratorD(testList, 2);
//        normalizer.fit(traindata);
//        normalizer1.fit(testdata); //Perform training for 2 epochs
//        traindata.setPreProcessor(normalizer);
//        testdata.setPreProcessor(normalizer1);

//        UIServer uiServer = UIServer.getInstance();
//        StatsStorage statsStorage = new InMemoryStatsStorage();
//        uiServer.attach(statsStorage);
//        UIListener l = UIListener.builder(new File("log.txt"))
//                //Plot loss curve, at every iteration (enabled and set to 1 by default)
//                .plotLosses(1)
//                //Plot the training set evaluation metrics: accuracy and f1 score
//                .trainEvaluationMetrics("softmax6", 0, Evaluation.Metric.ACCURACY, Evaluation.Metric.F1)
//                //Plot the parameter to update:ratios for each parameter, every 10 iterations
//                .updateRatios(10)
//                .build();
//        ScoreListener sl = new ScoreListener(10) {
//            @Override
//            public void iterationDone(SameDiff sd, At at, org.nd4j.linalg.dataset.api.MultiDataSet dataSet, Loss loss) {
//                super.iterationDone(sd, at, dataSet, loss);
////                System.out.println(loss);
//            }
//
//            @Override
//            public ListenerResponse epochEnd(SameDiff sd, At at, LossCurve lossCurve, long epochTimeMillis) {
//                System.out.println("Epoch " + at.epoch() + " loss: " + lossCurve.meanLoss(at.epoch()));
//                return super.epochEnd(sd, at, lossCurve, epochTimeMillis);
//
//            }
//        };

        int numEpochs = 150;
//        for (int i = 1; i < numEpochs; i++) {
        HistoryListener hl = new HistoryListener(config) {
            @Override
            public ListenerResponse validationDoneEvaluations(SameDiff sd, At at, long validationTimeMillis, EvaluationRecord evaluations) {
                System.out.println("Epoch:" + at.epoch() + " " + evaluations.evaluation("softmax6"));
                return super.validationDoneEvaluations(sd, at, validationTimeMillis, evaluations);
            }
        };
//        sd.setListeners(new Listener[]{l});
        double acc = 0.00;
        File saveFileForInference = new File("GCNModel.sabbih");
        for (int i = 0; i < numEpochs; i++) {


            History history = sd.fit(traindata, 1);

//        System.out.println(history.trainingEval(Evaluation.Metric.ACCURACY));

            //Evaluate on test set:
            String outputVariable = "softmax6";
            Evaluation evaluation = new Evaluation();
            sd.evaluate(testdata, outputVariable, 0, new Evaluation[]{evaluation});

            //Print evaluation statistics:
            System.out.println("Epoch: " + i);
            System.out.println(evaluation.stats());
            System.out.println(history.lossCurve().meanLoss(0));

            if (evaluation.accuracy() > acc) {
                //save
                //Save the trained network for inference - FlatBuffers format
                sd.asFlatFile(saveFileForInference);
                acc = evaluation.accuracy();
                System.out.println("SAVING model at an accuracy of: " + evaluation.accuracy());
            }
            evaluation.accuracy();

        }

        System.out.println("the highest accuracy was: " + acc);
    }

}

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.layers.BaseLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.AdaDelta;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

public class MessagePassingImplementation extends BaseLayer<MessagePassingLayer> {

    public MessagePassingImplementation(NeuralNetConfiguration conf, DataType dataType) {
        super(conf, dataType);
    }

    public boolean isPretrainLayer() {
        return false;
    }

    @Override
    public INDArray activate(INDArray input, boolean training, LayerWorkspaceMgr workspaceMgr) {

        INDArray output = new NDArray();//preOutput(training, workspaceMgr);// this is a matrix of the updated feature vectors of each vertex

        long nRows = input.shape()[0];
        long featureSize = input.shape()[1];
        List<INDArray> vertexFeatures = new LinkedList<>();

        for (long i = 0; i < nRows; i++) {
            vertexFeatures.add(input.getRow(i));
        }
        IActivation messageActivation = ((MessagePassingLayer) conf.getLayer()).getMessageActivationFunction();
//        conf.get
//        Hv = Hv.add(message);
//        conf.

        // Configure message passing model
        MultiLayerConfiguration messageModel = new NeuralNetConfiguration.Builder()
                .seed(1010)
                .weightInit(WeightInit.XAVIER)
                .updater(new AdaDelta()) // try other updaters later
                .list()
                .layer(new DenseLayer.Builder().nIn(featureSize).nOut(featureSize)
                        .activation(messageActivation)
                        .build())
//                .layer(new DropoutLayer.Builder().dropOut(0.5).nIn(featureSize).nOut(featureSize).build())// Just for test
//                .layer(new DenseLayer.Builder().nIn(numNodes).nOut(numNodes)
//                        .activation(Activation.RELU)
//                        .build())
                .layer(new OutputLayer.Builder()
                        .activation(Activation.IDENTITY)
                        .nIn(featureSize).nOut(featureSize).build())
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(messageModel);
        net.init();
        // Configure update  model
        MultiLayerConfiguration updateModel = new NeuralNetConfiguration.Builder()
                .seed(1010)
                .weightInit(WeightInit.XAVIER)
                .updater(new AdaDelta()) // try other updaters later
                .list()
                .layer(new DenseLayer.Builder().nIn(featureSize).nOut(featureSize)
                        .activation(messageActivation)
                        .build())
                .layer(new OutputLayer.Builder()
                        .activation(Activation.IDENTITY)
                        .nIn(featureSize).nOut(featureSize).build())
                .build();
        MultiLayerNetwork net1 = new MultiLayerNetwork(updateModel);
        net1.init();

        vertexFeatures.parallelStream().forEach((data) -> {
                    //                    System.out.println("features");
//                    System.out.println(data_30.getFeatures());
                    INDArray Hv = Nd4j.zeros(data.columns());

                    // get neighbors of v
                    ArrayList<Integer> neighbors = new ArrayList<>();
                    for (int j = 0; j < data.columns() - 25; j++) {
                        if (data.getDouble(j) > 0) {
                            if (j != vertexFeatures.indexOf(data))
                                neighbors.add(j);
                        }
                    }
                    // message passing phase
                    for (Integer n : neighbors) {
                        List<INDArray> activations = net.feedForward(vertexFeatures.get(n), true);
                        INDArray message = activations.get(activations.size() - 1);
                        Hv = Hv.add(message);
                    }
                    // Update Phase
                    Hv.add(data); // add the hv of the vertex
//            DataSet set1 = new DataSet(Hv, Hv);
                    List<INDArray> activations = net1.feedForward(Hv);
                    INDArray update = activations.get(activations.size() - 1);
                    output.putRow(vertexFeatures.indexOf(data), update);

                }
        );


        return output;
    }


}

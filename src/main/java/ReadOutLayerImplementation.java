import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.layers.BaseLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;

import java.util.LinkedList;
import java.util.List;

public class ReadOutLayerImplementation extends BaseLayer<ReadoutLayer> {
    public ReadOutLayerImplementation(NeuralNetConfiguration conf, DataType dataType) {
        super(conf, dataType);
    }

    @Override
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
        MultiLayerConfiguration updateModel = new NeuralNetConfiguration.Builder()
                .seed(1010)
                .weightInit(WeightInit.XAVIER)
                .updater(new Nesterovs()) // try other updaters later
                .list()
                .layer(new LSTM.Builder().activation(messageActivation).nIn(1).nOut(2028).build())
//                .layer(new DropoutLayer.Builder().dropOut(0.5).nIn(featureSize).nOut(featureSize).build())// Just for test
//                .layer(new DenseLayer.Builder().nIn(numNodes).nOut(numNodes)
//                        .activation(Activation.RELU)
//                        .build())
                .layer(new RnnOutputLayer.Builder()
                        .activation(messageActivation).nIn(2028).nOut(2028).build())
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(updateModel);
        net.init();

        INDArray Ge = Nd4j.zeros(2028);
        vertexFeatures.parallelStream().reduce(Ge, (ge, data) -> {

            List<INDArray> activations = net.feedForward(data, true);
            return ge.add(activations.get(activations.size() - 1));
        }, INDArray::add);

        return Ge;
    }
}

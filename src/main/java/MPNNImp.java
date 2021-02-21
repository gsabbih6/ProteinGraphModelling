import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class MPNNImp {
    /*Build the mpnn model with predefine layers for binary classification*/
    private MultiLayerNetwork mpnn(long nRows, long nCol,long sampledFeatures) {
        int seed = 123;
        double learningRate = 0.5;
        int out=2; // binary
        //Configure the layer with custom GCN and Readout layers
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .weightInit(WeightInit.XAVIER)
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
                .layer(new ReadoutLayer.Builder().readActivationFunction(Activation.SIGMOID).nIn(nCol * nRows).nOut(sampledFeatures)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.SOFTMAX)
                        .nIn(sampledFeatures).nOut(out).build())
                .build();
        return new MultiLayerNetwork(conf);
    }
}

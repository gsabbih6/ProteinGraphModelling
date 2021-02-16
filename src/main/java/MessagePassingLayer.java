import lombok.Data;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.nn.conf.memory.MemoryReport;
import org.deeplearning4j.nn.params.ConvolutionParamInitializer;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collection;
import java.util.Map;

@Data

/*
 * The class is used to configure the Message Passing Layer. In order words it is reposible for setting up the
 * attributes for each Message passing Layer
 * */
public class MessagePassingLayer extends FeedForwardLayer {
    private IActivation messageActivationFunction;
    private IActivation updateActivationFunction;

    public MessagePassingLayer() {
    }

    public MessagePassingLayer(Builder builder) {
        super(builder);
        this.messageActivationFunction = builder.messageActivationFunction;
        this.updateActivationFunction = builder.updateActivationFunction;

        new DenseLayer.Builder().build();

    }

    @Override
    public Layer instantiate(NeuralNetConfiguration conf, Collection<TrainingListener> iterationListeners,
                             int layerIndex, INDArray layerParamsView, boolean initializeParams, DataType networkDType) {
        //The instantiate method is how we go from the configuration class (i.e., this class) to the implementation class
        // (i.e., a CustomLayerImpl instance)
        //For the most part, it's the same for each type of layer

        MessagePassingImplementation myCustomLayer = new MessagePassingImplementation(conf, networkDType);
        myCustomLayer.setListeners(iterationListeners);             //Set the iteration listeners, if any
        myCustomLayer.setIndex(layerIndex);                         //Integer index of the layer

        //Parameter view array: In Deeplearning4j, the network parameters for the entire network (all layers) are
        // allocated in one big array. The relevant section of this parameter vector is extracted out for each layer,
        // (i.e., it's a "view" array in that it's a subset of a larger array)
        // This is a row vector, with length equal to the number of parameters in the layer
        myCustomLayer.setParamsViewArray(layerParamsView);

        //Initialize the layer parameters. For example,
        // Note that the entries in paramTable (2 entries here: a weight array of shape [nIn,nOut] and biases of shape [1,nOut]
        // are in turn a view of the 'layerParamsView' array.
        Map<String, INDArray> paramTable = initializer().init(conf, layerParamsView, initializeParams);
        myCustomLayer.setParamTable(paramTable);
        myCustomLayer.setConf(conf);
        return myCustomLayer;
    }

    @Override
    public ParamInitializer initializer() {
        //This method returns the parameter initializer for this type of layer
        //In this case, we can use the DefaultParamInitializer, which is the same one used for DenseLayer
        //For more complex layers, you may need to implement a custom parameter initializer
        //See the various parameter initializers here:
        //https://github.com/eclipse/deeplearning4j/tree/master/deeplearning4j/deeplearning4j-nn/src/main/java/org/deeplearning4j/nn/params

        return ConvolutionParamInitializer.getInstance();
    }

    @Override
    public LayerMemoryReport getMemoryReport(InputType inputType) {
        //Memory report is used to estimate how much memory is required for the layer, for different configurations
        //If you don't need this functionality for your custom layer, you can return a LayerMemoryReport
        // with all 0s, or

        //This implementation: based on DenseLayer implementation
        InputType outputType = getOutputType(-1, inputType);

        long numParams = initializer().numParams(this);
        int updaterStateSize = (int) getIUpdater().stateSize(numParams);

        int trainSizeFixed = 0;
        int trainSizeVariable = 0;
        if (getIDropout() != null) {
            //Assume we dup the input for dropout
            trainSizeVariable += inputType.arrayElementsPerExample();
        }

        //Also, during backprop: we do a preOut call -> gives us activations size equal to the output size
        // which is modified in-place by activation function backprop
        // then we have 'epsilonNext' which is equivalent to input size
        trainSizeVariable += outputType.arrayElementsPerExample();

        return new LayerMemoryReport.Builder(layerName, MessagePassingLayer.class, inputType, outputType)
                .standardMemory(numParams, updaterStateSize)
                .workingMemory(0, 0, trainSizeFixed, trainSizeVariable)     //No additional memory (beyond activations) for inference
                .cacheMemory(MemoryReport.CACHE_MODE_ALL_ZEROS, MemoryReport.CACHE_MODE_ALL_ZEROS) //No caching in DenseLayer
                .build();
    }

    public static class Builder extends FeedForwardLayer.Builder {

        private IActivation messageActivationFunction;
        private IActivation updateActivationFunction;

        //        public Builder secondActivationFunction(String secondActivationFunction) {
//            return secondActivationFunction(Activation.fromString(secondActivationFunction));
//        }
        public Builder messageActivationFunction(Activation secondActivationFunction) {
            this.messageActivationFunction = secondActivationFunction.getActivationFunction();
            return this;
        }

        public Builder updateActivationFunction(Activation secondActivationFunction) {
            this.updateActivationFunction = secondActivationFunction.getActivationFunction();
            return this;
        }

        @Override
        @SuppressWarnings("unchecked")  //To stop warnings about unchecked cast. Not required.
        public MessagePassingLayer build() {
            return new MessagePassingLayer(this);
        }
    }
}

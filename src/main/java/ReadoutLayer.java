import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.ParamInitializer;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.util.Collection;
import java.util.Map;

public class ReadoutLayer extends FeedForwardLayer {
    private final IActivation readActivation;

    public ReadoutLayer(Builder builder) {
        this.readActivation=builder.readOutActivationFunction;
    }

    @Override
    public Layer instantiate(NeuralNetConfiguration neuralNetConfiguration, Collection<TrainingListener> collection, int i, INDArray indArray, boolean b, DataType dataType) {
        //The instantiate method is how we go from the configuration class (i.e., this class) to the implementation class
        // (i.e., a CustomLayerImpl instance)
        //For the most part, it's the same for each type of layer

        ReadOutLayerImplementation myCustomLayer = new ReadOutLayerImplementation(neuralNetConfiguration, dataType);
        myCustomLayer.setListeners(collection);             //Set the iteration listeners, if any
        myCustomLayer.setIndex(i);                         //Integer index of the layer

        //Parameter view array: In Deeplearning4j, the network parameters for the entire network (all layers) are
        // allocated in one big array. The relevant section of this parameter vector is extracted out for each layer,
        // (i.e., it's a "view" array in that it's a subset of a larger array)
        // This is a row vector, with length equal to the number of parameters in the layer
        myCustomLayer.setParamsViewArray(indArray);

        //Initialize the layer parameters. For example,
        // Note that the entries in paramTable (2 entries here: a weight array of shape [nIn,nOut] and biases of shape [1,nOut]
        // are in turn a view of the 'layerParamsView' array.
        Map<String, INDArray> paramTable = initializer().init(neuralNetConfiguration, indArray, b);
        myCustomLayer.setParamTable(paramTable);
        myCustomLayer.setConf(neuralNetConfiguration);
        return myCustomLayer;
    }

    @Override
    public ParamInitializer initializer() {
        return null;
    }

    @Override
    public LayerMemoryReport getMemoryReport(InputType inputType) {
        return null;
    }
    public static class Builder extends FeedForwardLayer.Builder {

        private IActivation readOutActivationFunction;

        public ReadoutLayer.Builder readActivationFunction(Activation readActivationFunction) {
            this.readOutActivationFunction = readActivationFunction.getActivationFunction();
            return this;
        }

        @Override
        @SuppressWarnings("unchecked")  //To stop warnings about unchecked cast. Not required.
        public ReadoutLayer build() {
            return new ReadoutLayer(this);
        }
    }
}

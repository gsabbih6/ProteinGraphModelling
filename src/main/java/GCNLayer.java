//import org.deeplearning4j.nn.api.Layer;
//import org.deeplearning4j.nn.api.ParamInitializer;
//import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
//import org.deeplearning4j.nn.conf.inputs.InputType;
//import org.deeplearning4j.nn.conf.layers.FeedForwardLayer;
//import org.deeplearning4j.nn.conf.memory.LayerMemoryReport;
//import org.deeplearning4j.nn.params.DefaultParamInitializer;
//import org.deeplearning4j.optimize.api.TrainingListener;
//import org.nd4j.linalg.activations.Activation;
//import org.nd4j.linalg.activations.IActivation;
//import org.nd4j.linalg.api.buffer.DataType;
//import org.nd4j.linalg.api.ndarray.INDArray;
//
//import java.util.Collection;
//import java.util.Map;
//
//public class GCNLayer extends FeedForwardLayer {
//    private IActivation secondActivationFunction;
//
//    public GCNLayer() {
//    }
//
//    private GCNLayer(Builder builder) {
//        super(builder);
//        this.secondActivationFunction = builder.secondActivationFunction;
//    }
//    public IActivation getSecondActivationFunction() {
//        //We also need setter/getter methods for our layer configuration fields (if any) for JSON serialization
//        return secondActivationFunction;
//    }
//    public void setSecondActivationFunction(IActivation secondActivationFunction) {
//        //We also need setter/getter methods for our layer configuration fields (if any) for JSON serialization
//        this.secondActivationFunction = secondActivationFunction;
//    }
//
//    @Override
//    public Layer instantiate(NeuralNetConfiguration conf, Collection<TrainingListener> collection, int i,
//                             INDArray indArray, boolean b, DataType networkDType) {
//        //The instantiate method is how we go from the configuration class (i.e., this class) to the implementation class
//        // (i.e., a CustomLayerImpl instance)
//        //For the most part, it's the same for each type of layer
//
//       GCNLayerImplementation myCustomLayer = new GCNLayerImplementation(conf, networkDType);
//        myCustomLayer.setListeners(collection);             //Set the iteration listeners, if any
//        myCustomLayer.setIndex(i);                         //Integer index of the layer
//
//        //Parameter view array: In Deeplearning4j, the network parameters for the entire network (all layers) are
//        // allocated in one big array. The relevant section of this parameter vector is extracted out for each layer,
//        // (i.e., it's a "view" array in that it's a subset of a larger array)
//        // This is a row vector, with length equal to the number of parameters in the layer
//        myCustomLayer.setParamsViewArray(indArray);
//
//        //Initialize the layer parameters. For example,
//        // Note that the entries in paramTable (2 entries here: a weight array of shape [nIn,nOut] and biases of shape [1,nOut]
//        // are in turn a view of the 'layerParamsView' array.
//        Map<String, INDArray> paramTable = initializer().init(conf, indArray, b);
//        myCustomLayer.setParamTable(paramTable);
//        myCustomLayer.setConf(conf);
//        return myCustomLayer;
//    }
//
//    @Override
//    public ParamInitializer initializer() {
//        return DefaultParamInitializer.getInstance();
//    }
//
//    @Override
//    public LayerMemoryReport getMemoryReport(InputType inputType) {
//        return new LayerMemoryReport();
//    }
//
//    public static class Builder extends FeedForwardLayer.Builder<Builder> {
//
//        private IActivation secondActivationFunction;
//
//        //This is an example of a custom property in the configuration
//
//        /**
//         * A custom property used in this custom layer example. See the README.md for details
//         *
//         * @param secondActivationFunction Second activation function for the layer
//         */
//        public Builder secondActivationFunction(String secondActivationFunction) {
//            return secondActivationFunction(Activation.fromString(secondActivationFunction));
//        }
//
//        /**
//         * A custom property used in this custom layer example. See the README.md for details
//         *
//         * @param secondActivationFunction Second activation function for the layer
//         */
//        public Builder secondActivationFunction(Activation secondActivationFunction) {
//            this.secondActivationFunction = secondActivationFunction.getActivationFunction();
//            return this;
//        }
//
//        @Override
//        @SuppressWarnings("unchecked")  //To stop warnings about unchecked cast. Not required.
//        public GCNLayer build() {
//            return new GCNLayer(this);
//        }
//    }
//}

import org.deeplearning4j.exception.DL4JInvalidInputException;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.gradient.DefaultGradient;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.layers.BaseLayer;
import org.deeplearning4j.nn.params.DefaultParamInitializer;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;

public class ReadOutLayerImplementation extends BaseLayer<ReadoutLayer> {
    public ReadOutLayerImplementation(NeuralNetConfiguration conf, DataType dataType) {
        super(conf, dataType);
    }

    @Override
    public boolean isPretrainLayer() {
        return false;
    }

    @Override
    public INDArray activate(boolean training, LayerWorkspaceMgr workspaceMgr) {
        INDArray output = preOutput(training, workspaceMgr);// this is a matrix of the updated feature vectors of each vertex
        IActivation readActivationFunction = ((ReadoutLayer) conf.getLayer()).getReadActivationFunction();
        INDArray ret1 = readActivationFunction.getActivation(output, training);
        INDArray ret = workspaceMgr.createUninitialized(ArrayType.ACTIVATIONS, output.dataType(), new long[]{1, output.size(1)});

//        if (this.maskArray != null) {
//            this.applyMask(ret1);
//        }
        ret1.sum(ret,0);
        return ret1;
    }

    @Override
    protected INDArray preOutput(boolean training, LayerWorkspaceMgr workspaceMgr) {
        INDArray W = this.getParamWithNoise("W", training, workspaceMgr);
        INDArray b = this.getParamWithNoise("b", training, workspaceMgr);
        INDArray input = getInput().castTo(this.dataType);
        if (input.rank() == 2 && input.columns() == W.rows()) {
            INDArray ret = workspaceMgr.createUninitialized(ArrayType.ACTIVATIONS, W.dataType(), new long[]{input.size(0), W.size(1)});
            INDArray ret1 = workspaceMgr.createUninitialized(ArrayType.ACTIVATIONS, W.dataType(), new long[]{1, W.size(1)});
            input.castTo(ret.dataType()).mmuli(W, ret);
            if (this.hasBias()) {
                ret.addiRowVector(b);
            }
//            if (this.maskArray != null) {
//                this.applyMask(ret1);
//            }
            ret.sum(ret1, 0);
            return ret1;
        } else if (input.rank() != 2) {
            throw new DL4JInvalidInputException("Input that is not a matrix; expected matrix (rank 2), got rank " + input.rank() + " array with shape " + Arrays.toString(input.shape()) + ". Missing preprocessor or wrong input type? " + this.layerId());
        } else {
            throw new DL4JInvalidInputException("Input size (" + input.columns() + " columns; shape = " + Arrays.toString(input.shape()) + ") is invalid: does not match layer input size (layer # inputs = " + W.size(0) + ") " + this.layerId());
        }
//        System.out.println(input.sum(0));
//        input.sum(ret1,0);
//        System.out.println(sumOfColumns);
//        return ret1;//Nd4j.zeros(input.rows(),input.columns());
    }

    @Override
    public Pair<Gradient, INDArray> backpropGradient(INDArray epsilon, LayerWorkspaceMgr workspaceMgr) {
        /*
        The baockprop gradient method here is very similar to the BaseLayer backprop gradient implementation
        The only major difference is the two activation functions we have added in this example.
        Note that epsilon is dL/da - i.e., the derivative of the loss function with respect to the activations.
        It has the exact same shape as the activation arrays (i.e., the output of preOut and activate methods)
        This is NOT the 'delta' commonly used in the neural network literature; the delta is obtained from the
        epsilon ("epsilon" is dl4j's notation) by doing an element-wise product with the activation function derivative.
        Note the following:
        1. Is it very important that you use the gradientViews arrays for the results.
           Note the gradientViews.get(...) and the in-place operations here.
           This is because DL4J uses a single large array for the gradients for efficiency. Subsets of this array (views)
           are distributed to each of the layers for efficient backprop and memory management.
        2. The method returns two things, as a Pair:
           (a) a Gradient object (essentially a Map<String,INDArray> of the gradients for each parameter (again, these
               are views of the full network gradient array)
           (b) an INDArray. This INDArray is the 'epsilon' to pass to the layer below. i.e., it is the gradient with
               respect to the input to this layer
        */

        INDArray activationDerivative = preOutput(true, workspaceMgr);
        INDArray hstack = Nd4j.vstack(epsilon,Nd4j.zeros(input.rows()-1,epsilon.columns()));
        IActivation activation = ((ReadoutLayer) conf.getLayer()).getReadActivationFunction();
        activation.backprop(activationDerivative, epsilon);

        //The remaining code for this method: just copy & pasted from BaseLayer.backpropGradient
//        INDArray delta = epsilon.muli(activationDerivative);
        if (maskArray != null) {
            activationDerivative.muliColumnVector(maskArray);
        }

        Gradient ret = new DefaultGradient();

        INDArray weightGrad = gradientViews.get(DefaultParamInitializer.WEIGHT_KEY);    //f order
        INDArray vstack = Nd4j.vstack(activationDerivative,Nd4j.zeros(input.rows()-1,input.columns()));
//System.out.println(activationDerivative.shape()[0]);
        Nd4j.gemm(input, vstack, weightGrad, true, false, 1.0, 0.0);
        INDArray biasGrad = gradientViews.get(DefaultParamInitializer.BIAS_KEY);
        biasGrad.assign(activationDerivative.sum(0));  //TODO: do this without the assign

        ret.gradientForVariable().put(DefaultParamInitializer.WEIGHT_KEY, weightGrad);
        ret.gradientForVariable().put(DefaultParamInitializer.BIAS_KEY, biasGrad);

        INDArray epsilonNext = params.get(DefaultParamInitializer.WEIGHT_KEY).mmul(activationDerivative.transpose()).transpose();

        return new Pair<>(ret, workspaceMgr.leverageTo(ArrayType.ACTIVATION_GRAD, epsilonNext));
    }

}

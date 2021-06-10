import org.deeplearning4j.exception.DL4JInvalidInputException;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.layers.BaseLayer;
import org.deeplearning4j.nn.layers.BaseOutputLayer;
import org.deeplearning4j.nn.workspace.ArrayType;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.common.primitives.Pair;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.BaseOpContext;
import org.nd4j.linalg.api.ops.impl.transforms.custom.LayerNorm;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

public class CustomOutputImplementation extends BaseOutputLayer<CustomOutlayer> {
    public CustomOutputImplementation(NeuralNetConfiguration conf, DataType dataType) {
        super(conf, dataType);
    }

    @Override
    protected INDArray getLabels2d(LayerWorkspaceMgr layerWorkspaceMgr, ArrayType arrayType) {
        return null;
    }

    @Override
    public boolean isPretrainLayer() {
        return false;
    }
    public INDArray activate(boolean training, LayerWorkspaceMgr workspaceMgr) {
        INDArray z = this.preOutput(training, workspaceMgr);
        INDArray ret = this.layerConf().getActivationFn().getActivation(z, training);
//        if (this.maskArray != null) {
//            this.applyMask(ret);
//        }

        return ret;
    }
    @Override
    protected INDArray preOutput(boolean training, LayerWorkspaceMgr workspaceMgr) {
//        this.assertInputSet(forBackprop);
        this.applyDropOutIfNecessary(training, workspaceMgr);
        INDArray W = this.getParamWithNoise("W", training, workspaceMgr);
        INDArray b = this.getParamWithNoise("b", training, workspaceMgr);
        INDArray g = this.hasLayerNorm() ? this.getParam("g") : null;
        INDArray input = this.input.castTo(this.dataType);
        if (input.rank() == 2 && input.columns() == W.rows()) {
            INDArray ret = workspaceMgr.createUninitialized(ArrayType.ACTIVATIONS, W.dataType(), new long[]{input.size(0), W.size(1)});
            input.castTo(ret.dataType()).mmuli(W, ret);
            INDArray preNorm = ret;
            if (this.hasLayerNorm()) {
//                preNorm = forBackprop ? ret.dup(ret.ordering()) : ret;
                Nd4j.getExecutioner().exec(new LayerNorm(preNorm, g, ret, true, new int[]{1}));
            }

            if (this.hasBias()) {
                ret.addiRowVector(b);
            }

//            if (this.maskArray != null) {
//                this.applyMask(ret);
//            }

            return ret;
        } else if (input.rank() != 2) {
            throw new DL4JInvalidInputException("Input that is not a matrix; expected matrix (rank 2), got rank " + input.rank() + " array with shape " + Arrays.toString(input.shape()) + ". Missing preprocessor or wrong input type? " + this.layerId());
        } else {
            throw new DL4JInvalidInputException("Input size (" + input.columns() + " columns; shape = " + Arrays.toString(input.shape()) + ") is invalid: does not match layer input size (layer # inputs = " + W.size(0) + ") " + this.layerId());
        }
    }
}

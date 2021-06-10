//import org.nd4j.linalg.api.ndarray.INDArray;
//import org.nd4j.linalg.dataset.api.DataSet;
//import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
//
//public class CustomDSITE  implements DataSetIterator {
//    private INDArray[] inputs,adjArray, desiredOutputs;
//    private int itPosition = 0; // the iterator position in the set.
//
//    public CustomDSITE(INDArray[] inputsArray,
//                       INDArray[] adjArray,
//                       INDArray[] desiredOutputsArray,
//                            int numSamples,
//                            int inputDim,
//                            int outputDim) {
//        inputs = inputsArray;
//        adjArray = adjArray;//Nd4j.create(desiredOutputsArray, new int[]{numSamples, outputDim});
//        desiredOutputs=desiredOutputsArray;
//    }
//
//    public DataSet next(int num) {
//        // get a view containing the next num samples and desired outs.
//        INDArray dsInput = inputs.get(
//                NDArrayIndex.interval(itPosition, itPosition + num),
//                NDArrayIndex.all());
//        INDArray dsDesired = desiredOutputs.get(
//                NDArrayIndex.interval(itPosition, itPosition + num),
//                NDArrayIndex.all());
//
//        itPosition += num;
//
//        return new DataSet(dsInput, dsDesired);
//    }